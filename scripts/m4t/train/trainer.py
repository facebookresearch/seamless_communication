# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any, Optional, Tuple, Dict, List

import os
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.nn.padding import PaddingMask
from fairseq2.optim.lr_scheduler import MyleLR
from m4t_scripts.train import dataloader, dist_utils
from torch.optim import Adam

from seamless_communication.models.unity import UnitYModel, UnitYT2UModel
from m4t_scripts.train.configs import TrainingParams

logger = logging.getLogger(__name__)


class UnitYTrainWrapper(nn.Module):
    """Convenience wrapper that does a forward pass
    and returns S2T and T2U logits"""

    def __init__(self, model: UnitYModel):
        super().__init__()
        self.model: UnitYModel = model
        if isinstance(self.model.t2u_model, UnitYT2UModel):
            self.t2u: UnitYT2UModel = self.model.t2u_model
        else:
            raise NotImplementedError(
                "Expand UnitYTrainWrapper supports only instances of UnitYT2UModel as t2u"
            )

    def forward(
        self, batch: dataloader.MultimodalSeqsBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass, computes S2T and T2U losses"""
        assert self.model.t2u_model is not None
        assert batch.speech_to_text.src_tokens is not None
        # s2t
        speech_padding_mask = PaddingMask(
            seq_lens=batch.speech_to_text.src_lengths,
            batch_seq_len=int(torch.max(batch.speech_to_text.src_lengths).item()),
        )
        speech_encoder_out, speech_encoder_padding_mask = self.model.encode_speech(
            seqs=batch.speech_to_text.src_tokens,
            padding_mask=speech_padding_mask,
        )
        assert batch.speech_to_text.prev_output_tokens is not None
        s2t_prev_out_tokens_padding_mask = PaddingMask(
            seq_lens=batch.speech_to_text.target_lengths,
            batch_seq_len=int(torch.max(batch.speech_to_text.target_lengths).item()),
        )
        text_decoder_out, text_decoder_padding_mask = self.model.decode(
            seqs=batch.speech_to_text.prev_output_tokens,
            padding_mask=s2t_prev_out_tokens_padding_mask,
            encoder_output=speech_encoder_out,
            encoder_padding_mask=speech_encoder_padding_mask,
        )
        text_logits = self.model.final_proj(text_decoder_out)
        # t2u
        (
            unit_encoder_out,
            unit_encoder_padding_mask,
        ) = self.t2u.encode(
            text_decoder_output=text_decoder_out,
            text_decoder_padding_mask=text_decoder_padding_mask,
        )
        t2u_prev_out_tokens_padding_mask = PaddingMask(
            seq_lens=batch.text_to_units.target_lengths,
            batch_seq_len=int(torch.max(batch.text_to_units.target_lengths).item()),
        )
        unit_decoder_out, _ = self.t2u.decode(
            seqs=batch.text_to_units.prev_output_tokens,
            padding_mask=t2u_prev_out_tokens_padding_mask,
            encoder_output=unit_encoder_out,
            encoder_padding_mask=unit_encoder_padding_mask,
        )
        unit_logits = self.model.t2u_model.final_proj(unit_decoder_out)
        return (text_logits, unit_logits)


class CalcLoss:
    """Calculates per-token negative log likelihood loss for S2T and T2U"""

    def __init__(
        self,
        label_smoothing: float,
        s2t_pad_idx: Optional[int],
        t2u_pad_idx: Optional[int],
        s2t_skip_langtok_loss: bool = False,
    ):
        self.label_smoothing = label_smoothing
        self.s2t_pad_idx = s2t_pad_idx
        self.t2u_pad_idx = t2u_pad_idx
        self.s2t_ignore_prefix_size = 1 if s2t_skip_langtok_loss else 0
        self.t2u_ignore_prefix_size = 1

    def __call__(
        self,
        batch: dataloader.MultimodalSeqsBatch,
        text_logits: torch.Tensor,
        unit_logits: torch.Tensor,
    ) -> torch.Tensor:
        assert batch.speech_to_text.target_lengths is not None
        s2t_numel = torch.sum(batch.speech_to_text.target_lengths).to(
            text_logits.device
        )
        s2t_loss = SequenceModelOutput(
            logits=text_logits, pad_idx=self.s2t_pad_idx
        ).compute_loss(
            targets=batch.speech_to_text.target_tokens.to(text_logits.device),
            ignore_prefix_size=self.s2t_ignore_prefix_size,
            label_smoothing=self.label_smoothing,
        )
        assert batch.text_to_units.target_lengths is not None
        s2u_numel = torch.sum(batch.text_to_units.target_lengths).to(unit_logits.device)
        s2u_loss = SequenceModelOutput(
            logits=unit_logits, pad_idx=self.t2u_pad_idx
        ).compute_loss(
            targets=batch.text_to_units.target_tokens.to(unit_logits.device),
            ignore_prefix_size=1,
            label_smoothing=self.label_smoothing,
        )
        return s2t_loss / s2t_numel + s2u_loss / s2u_numel


class LossCollector:
    """Aggregrates loss history across nodes"""

    def __init__(self, device: Optional[torch.device] = None, reduce_op: str = "avg"):
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


class UnitYTrainer:
    CHECKPOINT_BEST = "checkpoint_best.pt"

    def __init__(
        self,
        model: UnitYModel,
        params: TrainingParams,
        train_data_loader: dataloader.UnityDataLoader,
        eval_data_loader: Optional[dataloader.UnityDataLoader],
        chck_save_dir: str,
        device: torch.device,
    ):
        self.params = params
        self.device = device
        self.float_dtype = self._get_float_dtype(self.params.float_dtype)
        self.train_data_loader = train_data_loader
        self.eval_data_loader = eval_data_loader
        self.chck_save_dir = chck_save_dir

        assert model.t2u_model is not None
        self.calc_loss = CalcLoss(
            label_smoothing=self.params.label_smoothing,
            s2t_pad_idx=model.pad_idx,
            t2u_pad_idx=model.t2u_model.pad_idx,
        )
        self._try_load_checkpoint(model=model)
        self.model = self._wrap_model_for_trainining(model=model)

        # TODO: make tweakable
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.params.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-08,
            maximize=False,
            weight_decay=0.0,
            fused=True,
        )

        self.grad_scaler = torch.cuda.amp.GradScaler() if self.float_dtype == torch.float16 else None  # type: ignore

        # TODO: allow scheduler selection
        self.lr_scheduler = MyleLR(
            optimizer=self.optimizer,
            num_warmup_steps=self.params.warmup_steps,
            start_lr=self.params.start_learning_rate,
        )

        self.train_loss_hist = LossCollector(device=self.device)
        self.epoch_idx: int = 0
        self.update_idx: int = 0
        self.patience_left: int = self.params.patience
        self.last_eval_loss: Optional[float] = None
        self.best_eval_loss: Optional[float] = None
        self.is_best_state: bool = False
        self.batch_sizes: List[int] = []
        self.gpu_usage: List[float] = []

    def _try_load_checkpoint(self, model: torch.nn.Module):
        chck_path = self.get_best_checkpoint_path()
        if os.path.exists(chck_path):
            logger.info(f"Loading state dict from {chck_path}")
            state_dict = torch.load(chck_path)
            model.load_state_dict(state_dict)

    @classmethod
    def _get_float_dtype(cls, float_dtype: str) -> torch.dtype:
        if float_dtype == "fp16":
            return torch.float16
        elif float_dtype == "fp32":
            return torch.float32
        elif float_dtype == "bf16":
            return torch.bfloat16
        else:
            raise ValueError(f"Unkown dtype literal: {float_dtype}")

    def _reset_stats(self) -> None:
        self.train_loss_hist.reset()
        self.epoch_idx = 0
        self.update_idx = 0
        self.patience_left = self.params.patience
        self.last_eval_loss = None
        self.best_eval_loss = None
        self.is_best_state = False
        self._reset_log_stats()

    def _reset_log_stats(self) -> None:
        self.batch_sizes.clear()
        self.gpu_usage.clear()
        self.ts = time.time()
        self.last_update_idx = self.update_idx

    def _record_gpu_usage(self) -> None:
        gb = (torch.cuda.memory_reserved(self.device) >> 20) / 1024.0
        self.gpu_usage.append(gb)

    def _get_avg_bsz(self) -> float:
        """Avg training batch size"""
        return (
            sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0.0
        )

    def _get_ups(self) -> float:
        """Updates per second"""
        ts_delta = time.time() - self.ts
        return (self.update_idx - self.last_update_idx) / ts_delta

    def _get_avg_gpu_usage(self) -> float:
        return sum(self.gpu_usage) / len(self.gpu_usage) if self.gpu_usage else 0.0

    def _wrap_model_for_trainining(self, model: UnitYModel) -> nn.Module:
        wrapped_model = UnitYTrainWrapper(model=model)
        if not dist_utils.is_dist_initialized():
            return wrapped_model
        return nn.parallel.DistributedDataParallel(
            wrapped_model,
            device_ids=[dist_utils.get_local_rank()],
            find_unused_parameters=True,
        )

    def _update_eval_stats(self, eval_loss: float) -> None:
        self.last_eval_loss = eval_loss
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
        loss_hist = LossCollector(device=self.device)
        self.model.eval()
        with torch.no_grad():
            self.eval_data_loader.reset()
            for batch in self.eval_data_loader.iterate_batches():
                assert batch.speech_to_text.src_tokens is not None
                loss = self.calc_loss(batch, *self.model(batch))
                if loss.isnan():
                    logger.warning("Eval loss value is NaN, setting to inf")
                    loss_val = float("Inf")
                else:
                    loss_val = loss.item()
                self._release_memory(batch)
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
                f"last lr={self.lr_scheduler.get_last_lr()[0]:.2E} "
                f"bsz_avg={self._get_avg_bsz():.1f} "
                f"ups={self._get_ups():.2f} "
                f"gpu_avg={self._get_avg_gpu_usage():.2f}Gb"
            )
            self._reset_log_stats()

    def _train_step(self, batch: dataloader.MultimodalSeqsBatch) -> None:
        """Run one train step"""
        self.model.train()
        self.optimizer.zero_grad()
        tokens, units = self.model(batch)
        loss = self.calc_loss(batch, tokens, units)
        # peak of gpu usage
        self._record_gpu_usage()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()  # type: ignore
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        self.lr_scheduler.step()
        assert batch.speech_to_text.src_tokens is not None
        self.train_loss_hist.update(1, loss.item())
        self.batch_sizes.append(batch.speech_to_text.src_tokens.shape[0])
        self._train_step_log()
        self._release_memory(batch)

    def _release_memory(self, batch: dataloader.MultimodalSeqsBatch) -> None:
        """Explicitly release large memory consumers"""
        del batch

    def _strip_state_key_prefixes(self, key: str) -> str:
        """Removes state_dict keys prefixes associated with model wrappers"""
        to_strip = ["module.", "model."]
        for prefix in to_strip:
            if key.startswith(prefix):
                key = key[len(prefix):]
        return key

    def _get_state(self) -> Dict[str, Any]:
        model_state_dict = self.model.state_dict()
        model_state_dict = {
            self._strip_state_key_prefixes(key): value
            for key, value in model_state_dict.items()
        }
        return model_state_dict

    def _get_chck_path(self) -> str:
        ts = str(int(time.time()))
        epoch = str(self.epoch_idx).zfill(3)
        update = str(self.update_idx).zfill(6)
        eval_loss = f"{self.last_eval_loss:.4f}"
        name = f"{ts}_{epoch}_{update}_{eval_loss}.pt"
        return os.path.join(self.chck_save_dir, name)

    def _get_best_checkpoint_link_path(self) -> str:
        return os.path.join(self.chck_save_dir, self.CHECKPOINT_BEST)

    def get_best_checkpoint_path(self) -> str:
        return os.path.realpath(self._get_best_checkpoint_link_path())

    def _save_model(self):
        if dist_utils.is_main_process():
            state_dict = self._get_state()
            save_path = self._get_chck_path()
            logger.info(f"Saving checkpoint to {save_path}")
            torch.save(state_dict, save_path)
            if self.is_best_state:
                best_link_path = self._get_best_checkpoint_link_path()
                if os.path.exists(best_link_path):
                    os.unlink(best_link_path)
                os.symlink(save_path, best_link_path)
                logger.info(
                    f"Updating pointer to the best checkpoint {best_link_path} -> {save_path}"
                )
        if dist_utils.is_dist_initialized():
            dist.barrier()

    def run(self):
        logger.info("Start training")
        self._reset_stats()
        self._eval_model()
        while self.epoch_idx < self.params.max_epochs and self.patience_left:
            for train_batch in self.train_data_loader.iterate_batches():
                self._train_step(batch=train_batch)
                if self.update_idx and self.update_idx % self.params.eval_steps == 0:
                    self._eval_model()
                    if self.is_best_state:
                        self._save_model()
                    elif not self.patience_left:
                        no_improve_steps = self.params.eval_steps * self.params.patience
                        logger.info(
                            f"Early termination, as eval loss did not improve over last {no_improve_steps} updates"
                        )
                        break
                self.update_idx += 1
            self.train_data_loader.reset()
            self.epoch_idx += 1
