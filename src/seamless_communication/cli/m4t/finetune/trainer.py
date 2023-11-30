# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from fairseq2.data import VocabularyInfo
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.padding import PaddingMask
from fairseq2.optim.lr_scheduler import MyleLR
from fairseq2.typing import Device
from torch.optim import Adam

from seamless_communication.cli.m4t.finetune import dataloader, dist_utils
from seamless_communication.models.unity import UnitYModel

logger = logging.getLogger(__name__)


class FinetuneMode(Enum):
    SPEECH_TO_SPEECH = "SPEECH_TO_SPEECH"
    SPEECH_TO_TEXT = "SPEECH_TO_TEXT"
    TEXT_TO_SPEECH = "TEXT_TO_SPEECH"


@dataclass
class FinetuneParams:
    save_model_path: Path
    """Path were to save finetuned model."""

    finetune_mode: FinetuneMode = FinetuneMode.TEXT_TO_SPEECH
    """Allows to freeze S2T or T2U part of the model"""

    max_epochs: int = 10
    """ Maximum number of trainign epochs"""

    label_smoothing: float = 0.2
    """ Label smoothing coefficient for nll_loss """

    warmup_steps: int = 100
    """ Number of steps with linearly increasing LR"""

    log_steps: int = 10
    """ Log inner loss after each `log_steps` training steps"""

    eval_steps: int = 50
    """ Get eval loss after each `eval_steps` training steps """

    patience: int = 3
    """ Terminate if eval loss did not improve
    over the last `patience * eval_steps` training steps"""

    learning_rate: float = 1e-5
    """ Optimizer learining rate """

    train_batch_size: int = 5
    """The batch size during train steps"""

    eval_batch_size: int = 5
    """The batch size during evaluation."""

    device: Device = torch.device("cuda")
    """ Where to run computation"""


class UnitYFinetuneWrapper(nn.Module):
    """Convenience wrapper that does a forward pass
    and returns S2T and T2U logits"""

    def __init__(self, model: UnitYModel, mode: FinetuneMode, device: Device):
        super().__init__()
        assert model.t2u_model is not None
        self.model: UnitYModel = model
        self.freeze_s2t: bool = mode == FinetuneMode.TEXT_TO_SPEECH
        self.freeze_t2u: bool = mode == FinetuneMode.SPEECH_TO_TEXT
        self.device = device

    def forward(
        self, batch: dataloader.MultimodalSeqsBatch
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert self.model.t2u_model is not None
        dummy_context = contextmanager(lambda: iter([None]))()
        with torch.no_grad() if self.freeze_s2t else dummy_context:  # type:ignore
            assert batch.speech_to_text.src_tokens is not None
            seqs = batch.speech_to_text.src_tokens.to(self.device)
            seq_lens = batch.speech_to_text.src_lengths.to(self.device)
            speech_encoder_out, speech_encoder_padding_mask = self.model.encode_speech(
                seqs=seqs, padding_mask=PaddingMask(seq_lens, seqs.size(1))
            )
            assert batch.speech_to_text.prev_output_tokens is not None
            seqs = batch.speech_to_text.prev_output_tokens.to(self.device)
            seq_lens = batch.speech_to_text.target_lengths.to(self.device)
            text_decoder_out, text_decoder_padding_mask = self.model.decode(
                seqs=seqs,
                padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                encoder_output=speech_encoder_out,
                encoder_padding_mask=speech_encoder_padding_mask,
            )
            text_logits = self.model.final_proj(text_decoder_out)
        if batch.text_to_units.prev_output_tokens is None:
            return (text_logits, None)
        dummy_context = contextmanager(lambda: iter([None]))()
        with torch.no_grad() if self.freeze_t2u else dummy_context:  # type:ignore
            (
                unit_encoder_out,
                unit_encoder_padding_mask,
            ) = self.model.t2u_model.encode(
                text_decoder_output=text_decoder_out,
                text_decoder_padding_mask=text_decoder_padding_mask,
            )
            seqs = batch.text_to_units.prev_output_tokens.to(self.device)
            seq_lens = batch.text_to_units.target_lengths.to(self.device)
            unit_decoder_out, _ = self.model.t2u_model.decode(
                seqs=seqs,
                padding_mask=PaddingMask(seq_lens, seqs.size(1)),
                encoder_output=unit_encoder_out,
                encoder_padding_mask=unit_encoder_padding_mask,
            )
            unit_logits = self.model.t2u_model.final_proj(unit_decoder_out)

        return (text_logits, unit_logits)


class CalcLoss:
    """Calculates negative log likelihood loss for S2T and T2U"""

    def __init__(
        self,
        label_smoothing: float,
        s2t_vocab_info: VocabularyInfo,
        t2u_vocab_info: VocabularyInfo,
    ):
        self.label_smoothing = label_smoothing
        self.s2t_vocab_info = s2t_vocab_info
        self.t2u_vocab_info = t2u_vocab_info

    def __call__(
        self,
        batch: dataloader.MultimodalSeqsBatch,
        text_logits: torch.Tensor,
        unit_logits: Optional[torch.Tensor],
    ) -> torch.Tensor:
        assert batch.speech_to_text.target_lengths is not None
        s2t_numel = torch.sum(batch.speech_to_text.target_lengths).to(
            text_logits.device
        )
        s2t_loss = SequenceModelOutput(
            logits=text_logits, vocab_info=self.s2t_vocab_info
        ).compute_loss(
            targets=batch.speech_to_text.target_tokens.to(text_logits.device),
            ignore_prefix_size=1,
            label_smoothing=self.label_smoothing,
        )
        if unit_logits is None:
            return s2t_loss / s2t_numel
        assert batch.text_to_units.target_lengths is not None
        s2u_numel = torch.sum(batch.text_to_units.target_lengths).to(unit_logits.device)
        s2u_loss = SequenceModelOutput(
            logits=unit_logits, vocab_info=self.t2u_vocab_info
        ).compute_loss(
            targets=batch.text_to_units.target_tokens.to(unit_logits.device),
            ignore_prefix_size=1,
            label_smoothing=self.label_smoothing,
        )
        return s2t_loss / s2t_numel + s2u_loss / s2u_numel


class LossCollector:
    """Aggregrates loss history across nodes"""

    def __init__(self, device: Optional[Device] = None, reduce_op: str = "avg"):
        self.n_samples: float = 0
        self.val_sum: float = 0.0
        self.reduce_op = reduce_op
        self.device = device
        self.is_distributed = dist_utils.is_dist_initialized()

    def reset(self) -> None:
        self.n_samples = 0
        self.val_sum = 0.0

    def update(self, n_samples: int, batch_loss: float) -> None:
        self.n_samples += n_samples
        self.val_sum += batch_loss

    def reduce(self) -> float:
        n_samples, val_sum = self._collect()
        if self.reduce_op == "avg":
            return val_sum / (n_samples + 1)
        if self.reduce_op == "sum":
            return val_sum
        raise ValueError()

    def _collect(self) -> Tuple[float, float]:
        if not self.is_distributed:
            return self.n_samples, self.val_sum
        local_val = torch.tensor([[self.n_samples, self.val_sum]], device=self.device)
        all_vals = [
            torch.zeros((1, 2), device=self.device)
            for _ in range(dist_utils.get_world_size())
        ]
        dist.all_gather(all_vals, local_val)
        losses = torch.concat(all_vals, dim=0)
        reduced = torch.sum(losses, dim=0).reshape(2).cpu()
        return reduced[0].item(), reduced[1].item()


class UnitYFinetune:
    def __init__(
        self,
        model: UnitYModel,
        params: FinetuneParams,
        train_data_loader: dataloader.UnitYDataLoader,
        eval_data_loader: Optional[dataloader.UnitYDataLoader] = None,
    ):
        self.params = params

        assert model.t2u_model is not None
        self.calc_loss = CalcLoss(
            label_smoothing=self.params.label_smoothing,
            s2t_vocab_info=model.target_vocab_info,
            t2u_vocab_info=model.t2u_model.target_vocab_info,
        )
        self.model = self._wrap_model_for_trainining(model=model)
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.params.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-08,
            maximize=False,
            weight_decay=0.0,
            fused=True,
        )
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.lr_scheduler = MyleLR(
            optimizer=self.optimizer,
            num_warmup_steps=self.params.warmup_steps,
            start_lr=1e-9,
        )

        self.train_loss_hist = LossCollector(device=params.device)
        self.epoch_idx: int = 0
        self.update_idx: int = 0
        self.patience_left: int = self.params.patience
        self.best_eval_loss: Optional[float] = None
        self.is_best_state: bool = False

    def _reset_stats(self) -> None:
        self.train_loss_hist.reset()
        self.epoch_idx = 0
        self.update_idx = 0
        self.patience_left = self.params.patience
        self.best_eval_loss = None
        self.is_best_state = False

    def _wrap_model_for_trainining(self, model: UnitYModel) -> nn.Module:
        wrapped_model = UnitYFinetuneWrapper(
            model=model, mode=self.params.finetune_mode, device=self.params.device
        )
        if not dist_utils.is_dist_initialized():
            return wrapped_model
        return nn.parallel.DistributedDataParallel(
            wrapped_model,
            device_ids=[dist_utils.get_local_rank()],
            find_unused_parameters=True,
        )

    def _update_eval_stats(self, eval_loss: float) -> None:
        self.is_best_state = (
            self.best_eval_loss is None or eval_loss < self.best_eval_loss
        )
        self.best_eval_loss = eval_loss if self.is_best_state else self.best_eval_loss
        self.patience_left = (
            self.params.patience if self.is_best_state else self.patience_left - 1
        )
        logger.info(
            f"Eval after {self.update_idx} updates: "
            f"loss={eval_loss:.4f} "
            f"best_loss={self.best_eval_loss:.4f} "
            f"patience_steps_left={self.patience_left}"
        )

    def _eval_model(self) -> None:
        """Calc avg loss on eval dataset and update evaluation stats"""
        if self.eval_data_loader is None:
            return
        logger.info("Run evaluation")
        loss_hist = LossCollector(device=self.params.device)
        self.model.eval()
        with torch.no_grad():
            for batch in self.eval_data_loader.get_dataloader():
                assert batch.speech_to_text.src_tokens is not None
                loss = self.calc_loss(batch, *self.model(batch))
                if loss.isnan():
                    logger.warning("Eval loss value is NaN, setting to inf")
                    loss_val = float("Inf")
                else:
                    loss_val = loss.item()
                del batch  # force memory release
                loss_hist.update(1, loss_val)
        eval_loss = loss_hist.reduce()
        self._update_eval_stats(eval_loss)

    def _train_step_log(self):
        """Log train stats"""
        if (self.update_idx + 1) % self.params.log_steps == 0:
            avg_loss = self.train_loss_hist.reduce()
            self.train_loss_hist.reset()
            logger.info(
                f"Epoch {str(self.epoch_idx + 1).zfill(3)} / "
                f"update {str(self.update_idx + 1).zfill(5)}: "
                f"train loss={avg_loss:.4f} "
                f"last lr={self.lr_scheduler.get_last_lr()[0]:.2E}"
            )

    def _train_step(self, batch: dataloader.MultimodalSeqsBatch) -> None:
        """Run one train step"""
        self.model.train()
        self.optimizer.zero_grad()
        tokens, units = self.model(batch)
        loss = self.calc_loss(batch, tokens, units)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.lr_scheduler.step()
        assert batch.speech_to_text.src_tokens is not None
        self.train_loss_hist.update(1, loss.item())
        self._train_step_log()

    def _save_model(self):
        logger.info("Saving model")
        if dist_utils.is_main_process():
            state_dict = {
                key.replace("module.model.", ""): value
                for key, value in self.model.state_dict().items()
            }
            torch.save(state_dict, self.params.save_model_path)
        if dist_utils.is_dist_initialized():
            dist.barrier()

    def run(self):
        logger.info("Start finetuning")
        self._reset_stats()
        self._eval_model()
        batch_itr = self.train_data_loader.get_dataloader()
        while self.epoch_idx < self.params.max_epochs and self.patience_left:
            for train_batch in batch_itr:
                self._train_step(batch=train_batch)
                if self.update_idx and self.update_idx % self.params.eval_steps == 0:
                    self._eval_model()
                    if self.is_best_state:
                        self._save_model()
                    elif not self.patience_left:
                        no_improve_steps = self.params.eval_steps * self.params.patience
                        logger.info(
                            "Early termination, as eval loss did not improve "
                            f"over last {no_improve_steps} updates"
                        )
                        break
                self.update_idx += 1
            self.epoch_idx += 1
