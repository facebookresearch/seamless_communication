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
import pickle

import torch

from torch.optim import AdamW
from fairseq2.optim.lr_scheduler import MyleLR
from fairseq2.nn.padding import PaddingMask

from seamless_communication.cli.m4t.classification_head import dataloader
from seamless_communication.models.unity import UnitYModel
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
        "--label_smoothing",
        type=float,
        default=0.1,
        help=("Label smoothing"),
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


def plot_losslog(losslog: list, save_to: Path = None, yscale='log'):
    # TODO: Make this look good
    plt.plot(losslog)
    plt.yscale(yscale)
    plt.title('Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    if save_to:
        plt.savefig(save_to)
        plt.clf()
        with open(save_to.parent / "losslog.pkl", "wb") as f:
            pickle.dump(losslog, f)
    else:
        plt.show()


def train(head: torch.nn.Module,
          frozen_model: UnitYModel,
          dataloader: dataloader.UnitYLanguageIDDataLoader,
          params: ClassificationHeadTrainParams,
          label_smoothing: float = 0.1,
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
            
            # Run batches through train step
            for seqs, labels in tqdm(dataloader.get_dataloader(), desc="Training Steps"):
                assert seqs.src_tokens is not None
                optimizer.zero_grad()
                seqs.src_tokens = seqs.src_tokens.to(params.device)
                labels = labels.to(params.device)
                
                with torch.autocast(device_type=params.device.type, dtype=params.float_dtype):
                    mask = PaddingMask(seqs.src_lengths, seqs.src_tokens.size(1)).to(params.device)
                    vector, _ = frozen_model.encode(seqs.src_tokens, padding_mask=mask)
                    probs = head(vector)
                
                loss = torch.nn.functional.cross_entropy(probs, labels, label_smoothing=label_smoothing)
                if loss.isnan().any().item():
                    error = RuntimeError("Train loss is NaN! Something is wrong in the model!")
                    logger.error(seqs)
                    logger.error(error)
                    raise error
                
                losslog.append(loss.cpu().item())
                if len(losslog) % 5 == 0:
                    logger.info(f"Train Loss: {sum(losslog[-5:]) / 5}")
                    plot_losslog(losslog, save_to=params.save_model_path.parent / ".checkpoints/losslog.png")
                    
                    if len(losslog) % 2 == 0:
                        torch.save(head.state_dict(),
                            params.save_model_path.parent / f".checkpoints/checkpoint-{len(losslog)//10}.pt")
                
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                lr_scheduler.step()
    # Catch SIGINT (^C) keyboard interrupt, and save model before terminating
    except KeyboardInterrupt:
        logger.info("[SIGINT] Saving optimizer state. Exiting cleanly...")
        torch.save(optimizer.state_dict(), params.save_model_path / "optimizer_state.pth")

    # Log average loss over last 5 batches, this is more stable
    logger.info(f"Final Loss: {sum(losslog[-5:]) / 5}")
            
    # Return trained head and losslog as tensor
    # This makes it possible to do math operations easily
    return head, torch.tensor(losslog)


def main() -> None:
    args = init_parser().parse_args()
    device = torch.device(args.device)
    float_dtype = torch.float16 if torch.device(args.device).type != "cpu" else torch.bfloat16
    
    text_tokenizer = load_unity_text_tokenizer(args.model_name)
    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)
    
    # Freeze everything in the model, only train classification head
    model = load_unity_model(args.model_name, device=torch.device("cpu"), dtype=torch.float32)
    model.train()
    for _, module in model.named_modules():
        for param in module.parameters():
            param.requires_grad = False
    
    
    # TODO: Find embed dim from model
    head = ClassificationHead(model.model_dim, args.num_layers, args.num_languages)
    head.train()

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
        label_smoothing=args.label_smoothing,
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

    torch.save(trained_head.state_dict(), args.save_model_path)
    

if __name__ == "__main__":
    main()
