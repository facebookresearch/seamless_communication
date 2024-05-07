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

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from fairseq2.optim.lr_scheduler import MyleLR

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils, trainer
from seamless_communication.models.unity import (
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.cli.m4t.classification_head.load_classification_head import ClassificationHead

logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
)

logger = logging.getLogger("finetune")


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
        "--eval_dataset",
        type=Path,
        required=True,
        help="Path to manifest with eval samples",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="seamlessM4T_medium",
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
    )
    parser.add_argument(
        "--save_model_to",
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
        "--eval_steps",
        type=int,
        default=50,
        help=("Get eval loss after each `eval_steps` training steps "),
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
        help=("Log inner loss after each `log_steps` training steps"),
    )
    parser.add_argument(
        "--mode",
        type=trainer.FinetuneMode,
        choices=list(trainer.FinetuneMode),
        default=trainer.FinetuneMode.SPEECH_TO_TEXT,
        help=(
            "* `SPEECH_TO_SPEECH` -- finetune S2T and T2U parts of the model; "
            "* `TEXT_TO_SPEECH` -- finetune only T2U; "
            "* `SPEECH_TO_TEXT` -- finetune only S2T"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=("Device to fine-tune on. See `torch.device`."),
    )
    parser.add_argument(
        "--input_dim", 
        type=int, 
        default=768, 
        help="The size of the output from your model"
    )
    parser.add_argument(
        "--num_languages", 
        type=int, 
        default=2, 
        help="The number of classes you have"
    )
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=256, 
        help="The size of the hidden layer in the classification head"
    )
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=2, 
        help="The number of layers in the classification head"
    )
    return parser


def calc_loss():
    ...
    CrossEntropyLoss
    # return loss

def trainer(self,
            head,
            frozen_model,
            dataloader,
            params):
    
    logger.info("Start Training Language Head")
    dataloader = dataloader.get_dataloader()
    frozen_model.train()
    
    grad_scaler = torch.cuda.amp.GradScaler()
    optimizer = AdamW(
        params=frozen_model.parameters(),
        lr=params.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-08,
        maximize=False,
        weight_decay=0.0,
        fused=(params.device.type == "cuda"))
    lr_scheduler = MyleLR(
        optimizer=self.optimizer,
        num_warmup_steps=self.params.warmup_steps,
        start_lr=1e-9)

    losslog = list()
    # TODO: Implement training accoutrements: logging, capture interrupts etc
    for epoch in range(params.max_epochs):
        logger.info(f"Epoch {epoch}")
        for batch in tqdm(dataloader, desc="Training Steps"):
            # Run batch through train step
            optimizer.zero_grad()
            with torch.autocast(device_type=params.device.type, dtype=params.float_dtype):
                tokens, units = frozen_model.encode(batch)
            
            # Classification head
            _y = head(...)
            
            loss = calc_loss(batch, tokens, units)
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
    
    dist_utils.init_distributed([logger, trainer.logger])
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    # Freeze everything in the model, only train classification head
    for _, module in model.named_modules():
        for param in module.parameters():
            param.requires_grad = False

    classification_head = ClassificationHead(args.input_dim, args.num_languages, args.hidden_dim, args.num_layers)
    model.add_module('classification_head', classification_head)
    # TODO: add classification head layers to model
    
    # obj = torch.load(params.save_model_path)
    # classification_head.load_state_dict(obj)

    assert model.target_vocab_info == text_tokenizer.vocab_info
    
    if model.text_encoder is not None:
        model.text_encoder = None
    
    # Put model on selected device
    model = model.to(device)

    # Create daataloaders
    train_dataloader = dataloader.UnitYDataLoader(
        text_tokenizer=text_tokenizer,
        unit_tokenizer=unit_tokenizer,
        batching_config=dataloader.BatchingConfig(
            batch_size=args.batch_size,
            rank=dist_utils.get_rank(),
            world_size=dist_utils.get_world_size(),
            max_audio_length_sec=15.0,
            float_dtype=float_dtype,
        ),
        dataset_manifest_path=args.train_dataset)
    
    trained_head, losslog = trainer(
        head=classification_head,
        frozen_model=model,
        dataloader=train_dataloader,
        params=...
        # TODO: Create a class for parameters like FinetuneParams
    )
    
    # plot losslog
    # save trained head
    # torch.save(classification_head.state_dict(), params.save_model_path)
    

if __name__ == "__main__":
    main()
