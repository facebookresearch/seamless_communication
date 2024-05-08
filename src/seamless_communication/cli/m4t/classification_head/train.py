# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
from typing import List
from tqdm import tqdm
from dataclasses import dataclass
import matplotlib.pyplot as plt

import torch
from torch.optim import AdamW
from fairseq2.optim.lr_scheduler import MyleLR

from seamless_communication.models.unity import UnitYModel
from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.cli.m4t.classification_head.model import ClassificationHead

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("train_classification_head")

@dataclass
class ClassificationHeadTrainParams:
    save_model_path: Path
    
    float_dtype: torch.dtype
    
    max_epochs: int = 10
    """Maximum number of trainign epochs"""

    warmup_steps: int = 100
    """Number of steps with linearly increasing LR"""

    learning_rate: float = 1e-5
    """Optimizer learining rate"""

    batch_size: int = 5
    """The batch size during train steps"""

    device: torch.device = torch.device("cuda")
    """Where to run computation"""


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Example finetuning script for M4T models"
    )
    parser.add_argument(
        "--train_dataset",
        type=Path,
        required=True,
        help="Path to manifest with train samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--num_languages", 
        type=int, 
        help="The number of classes"
    )
    parser.add_argument(
        "--save_model_path",
        type=Path,
        required=True,
        help="Path to save best finetuned model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2343,
        help="Randomizer seed value",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help=(
            "Set early termination after `patience` number of evaluations "
            "without eval loss improvements"
        ),
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help=("Max number of training epochs"),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-7,
        help=("Finetuning learning rate"),
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help=("Number of steps with linearly increasing learning rate"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=2, 
        help="The number of layers in the classification head"
    )
    return parser


def trainer(head: torch.Module,
            frozen_model: UnitYModel,
            dataloader: dataloader.UnitYDataLoader,
            params: ClassificationHeadTrainParams,
            label_weights: torch.Tensor = None):
    
    logger.info("Start Training Language Head")
    
    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(
        params=head.parameters(),
        lr=params.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-08,
        maximize=False,
        weight_decay=0.0,
        fused=(params.device.type == "cuda"))
    lr_scheduler = MyleLR(
        optimizer=optimizer,
        num_warmup_steps=params.warmup_steps,
        start_lr=1e-9)

    losslog = list()
    try:
        for epoch in range(params.max_epochs):
            logger.info(f"Epoch {epoch}")
            
            _y = head(latent)
            
            loss = torch.nn.functional.cross_entropy(batch, _y, weight=label_weights)
            if loss.isnan().any().item():
                logger.error(batch.speech_to_text)
                raise RuntimeError("Train loss is NaN! Something is wrong in the model!")
            losslog.append(loss.item())
            
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()
            
            assert batch.speech_to_text.src_tokens is not None
            
    return head, losslog


def main() -> None:
    args = init_parser().parse_args()
    device = torch.device(args.device)
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    # Freeze everything in the model, only train classification head
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    for _, module in model.named_modules():
        for param in module.parameters():
            param.requires_grad = False

    classification_head = ClassificationHead(input_dim, args.num_languages, hidden_dim, args.num_layers)
    # TODO: based on base model, find what params to send here ^^^
    
    # model.add_module('classification_head', classification_head)
    # TODO: add classification head layers to model
    
    # obj = torch.load(params.save_model_path)
    # classification_head.load_state_dict(obj)

    assert model.target_vocab_info == text_tokenizer.vocab_info
    if model.text_encoder is not None:
        model.text_encoder = None
    
    # Put model on selected device
    model = model.to(device)
    head = head.to(device)

    # Create daataloaders
    train_dataloader = dataloader.UnitYLanguageIDDataLoader(
        num_languages=args.num_languages,
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=args.batch_size,
            max_audio_length_sec=15.0,
            float_dtype=float_dtype,
        ),
        dataset_manifest_path=args.train_dataset)
    
    trained_head, losslog = train(
        head=head,
        frozen_model=model,
        dataloader=train_dataloader,
        params=ClassificationHeadTrainParams(
            save_model_path=Path(args.save_model_path),
            float_dtype=float_dtype,
            max_epochs=args.max_epochs,
            warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=device
        )
    )

    # save trained head
    torch.save(trained_head.state_dict(), args.save_model_path)
    
    # plot losslog
    

if __name__ == "__main__":
    main()
