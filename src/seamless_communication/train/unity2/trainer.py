# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from pprint import pformat
from functools import partial
from typing import Any, Iterator, Literal, List, Optional, Union, final, Tuple, cast

import submitit
import torch
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel
from fairseq2 import setup_extensions
from fairseq2.checkpoint import FileCheckpointManager
from fairseq2.gang import Gang, ProcessGroupGang, FakeGang
from fairseq2.nn.transformer import (
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq2.metrics import (
    LogMetricRecorder,
    MetricRecorder,
    TensorBoardRecorder,
    record_metrics,
)
from fairseq2.data.text import TextTokenizer
from fairseq2.models.transformer import TransformerDecoderModel
from fairseq2.models.nllb import get_nllb_wrap_policy
from fairseq2.nn.fsdp import FSDPWrapPolicy, to_fsdp
from fairseq2.nn.utils.module import log_module, to_device
from fairseq2.optim import AdamW, DynamicLossScaler
from fairseq2.optim.lr_scheduler import LRScheduler, MyleLR, get_effective_lr
from fairseq2.typing import CPU
from fairseq2.utils.logging import setup_logging
from fairseq2.utils.profiler import Profiler, Stopwatch, log_environment_info
from fairseq2.utils.rng import RNGBag
from fairseq2.utils.state import StatefulObjectBag, FSDPOptimizerStateHandler
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.profiler import record_function
from seamless_communication.models.unity import (
    UnitYConfig,
    UnitTokenizer,
    UnitYModelOutput,
    load_unity_config,
    create_unity_model,
    load_unity_model,
)
from seamless_communication.models.unity.model import UnitYBatch
from seamless_communication.store import (
    load_gcmvn_stats,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.train.unity2.dataset import (
    UnitYDataset,
    load_unity_dataset,
)
from seamless_communication.train.unity2.metrics import UnitYMetricBag

logger = logging.getLogger(__name__)


@final
@dataclass
class UnitYTrainConfig:
    """Holds the configuration of an UnitY training job."""
    dataset_name: str
    """The dataset to train with."""

    tokenizer_name: str
    """The tokenizer to use."""

    model_config_or_name: UnitYConfig
    """The model configuration or name to train."""

    output_dir: Path
    """The output directory to store checkpoints and logs."""

    gcmvn_stats_name: str
    """The gcmvn stats to use."""

    gcmvn_prosody_input: bool = True
    """Whether to have gcmvn fbank as input to prosody encoder"""

    parallelism: Literal["none", "fsdp", "ddp"] = "fsdp"
    """The type of parallelism to use."""

    max_seq_len: int = 4000
    """The maximum source and target sequence lengths."""

    max_num_tokens: int = 8000
    """The maximum number of tokens per batch. This is effectively 4k as in fairseq1's batch_size
    as fairseq1 uses n_frames, here it uses fbank.shape"""

    shuffle_window_size: int = 100_000
    """The size of the sliding data shuffle window."""

    text_prefix_lang_tok: bool = False
    """Whether to prepend language token to the text input,
    currrently UnitY2 sets True, and ProsodyUnitY2 sets False"""

    num_prefetch: int = 10
    """The number of batches to prefetch in background."""

    dtype: str = "torch.float16"
    """The data type of the model."""

    lr: float = 0.0001
    """The initial (post-warm-up) learning rate for Adam."""

    max_grad_norm: Optional[float] = None
    """Maximal gradient norm, for gradient clipping. Values None or 0 result in no clipping."""

    num_lr_warmup_steps: int = 1000
    """The number of warm-up steps for the learning rate."""

    label_smoothing: float = 0.2
    """The amount of label smoothing to apply while computing the loss."""

    text_loss_weight: float = 8.0
    """The loss weight for text-decoder cross-entropy loss"""

    ignore_text_prefix_size: int = 0
    """Ignore text target prefix (for special prefix tokens)"""

    ignore_unit_prefix_size: int = 0
    """Ignore unit target prefix (for special prefix tokens)"""

    max_steps: int = 100_000
    """The maximum number of training steps."""

    gradient_accumulation: int = 4
    """The number of steps to accumulate gradients before an optimizer update."""

    validate_every_n_steps: int = 1000
    """The number of steps after which to validate the model."""

    checkpoint_every_n_steps: int = 1000
    """The number of steps after which to checkpoint."""

    save_model_every_n_steps: int = 1000
    """The number of steps after which to save a consolidated version of the model."""

    publish_metrics_every_n_steps: int = 10
    """The number of steps after which to publish training metrics."""

    seed: int = 1234
    """The RNG seed to use while starting the job."""

    profile: bool = False
    """If ``True``, runs the PyTorch profiler at the beginning of the training."""

    aux_loss_type: Optional[str] = None
    """the auxiliary loss type"""

    aux_loss_weight: float = 1.6
    """the auxiliary loss weight"""

    load_fairseq1_s2t_weight: Optional[str] = None
    """Holds the (M4T v2) S2T checkpoint path to initalize S2T model"""

    load_prosody_encoder_weight: Optional[str] = None
    """Holds the PRETSSEL checkpoint path to initialize prosody encoder weight"""


def unity_wrap_policy() -> Tuple[Optional[FSDPWrapPolicy], Optional[List[str]]]:
    """Return the FSDP wrap policy and ignored parameter names for ``arch_name``.

    :returns:
        - The FSDP wrap policy.
        - The ignored parameter names. Can contain regular expressions.
    """
    kls = (TransformerEncoder, TransformerDecoder)

    wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls=kls)

    return wrap_policy, None

class UnitYTrainer(StatefulObjectBag):
    config: UnitYTrainConfig
    model: Module
    dataset: UnitYDataset
    text_tokenizer: TextTokenizer
    unit_tokenizer: UnitTokenizer
    gang: Gang
    optimizer: Optimizer
    loss_scaler: DynamicLossScaler
    lr_scheduler: LRScheduler
    rng_bag: RNGBag
    step_nr: int
    train_metric_bag: UnitYMetricBag
    valid_metric_bag: UnitYMetricBag
    metric_recorders: List[MetricRecorder]
    profiler: Profiler
    stopwatch: Stopwatch

    def __init__(
        self,
        config: UnitYTrainConfig,
        model: Module,
        dataset: UnitYDataset,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: UnitTokenizer,
        gang: Gang,
        checkpoint_manager: FileCheckpointManager,
        rng_bag: RNGBag,
        stopwatch: Stopwatch,
    ) -> None:
        super().__init__()

        self.config = config

        self.dtype = eval(config.dtype)

        self.model = model

        self.dataset = dataset

        self.gcmvn_mean = self.gcmvn_std = None
        if config.gcmvn_prosody_input:
            _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(config.gcmvn_stats_name)
            self.gcmvn_mean = torch.tensor(_gcmvn_mean, dtype=self.dtype, device=gang.device)
            self.gcmvn_std = torch.tensor(_gcmvn_std, dtype=self.dtype, device=gang.device)

        self.data_iter = dataset.read(
            "train",
            text_tokenizer,
            unit_tokenizer,
            gang,
            self.dtype,
            config.max_seq_len,
            config.max_num_tokens,
            bucket_by_length=True,
            shuffle_window_size=config.shuffle_window_size,
            num_prefetch=config.num_prefetch,
            num_accumulate=config.gradient_accumulation,
            text_prefix_lang_tok=config.text_prefix_lang_tok,
            gcmvn_prosody_input=config.gcmvn_prosody_input,
            gcmvn_mean=self.gcmvn_mean,
            gcmvn_std=self.gcmvn_std,
        )

        self.text_tokenizer = text_tokenizer

        self.unit_tokenizer = unit_tokenizer

        self.gang = gang

        optimizer = AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            use_fp32=self.dtype == torch.float16,
            impl="fused" if config.parallelism != "fsdp" else "auto",
        )

        if config.parallelism != "fsdp":
            self.optimizer = optimizer
        else:
            self.register_stateful(
                "optimizer", optimizer, FSDPOptimizerStateHandler(model)
            )

        self.loss_scaler = DynamicLossScaler(
            optimizer,
            gang=gang,
            init_scale=128,
            min_scale=0.00001,
            gradient_accumulation=config.gradient_accumulation,
            enabled=self.dtype == torch.float16,
        )

        self.lr_scheduler = MyleLR(optimizer, config.num_lr_warmup_steps, start_lr=1e-7)

        self.rng_bag = rng_bag

        self.step_nr = 1

        if config.parallelism == "none":
            checkpoint_manager.replicated_keys = {"*"}
        elif config.parallelism == "ddp":
            # If we do not shard, save the model and the optimizer only on rank 0.
            checkpoint_manager.replicated_keys = {"model", "optimizer"}

        self.checkpoint_manager = checkpoint_manager

        self.train_metric_bag = UnitYMetricBag(gang)

        self.register_non_stateful("valid_metric_bag", UnitYMetricBag(gang))

        self.metric_recorders = [LogMetricRecorder(logger)]

        tb_dir = config.output_dir.joinpath("tb")

        if gang.rank == 0:
            self.metric_recorders.append(TensorBoardRecorder(tb_dir))

        self.profiler = Profiler(
            skip_first=150, active=3, log_dir=tb_dir, gang=gang, enabled=config.profile
        )

        self.stopwatch = stopwatch

    def run(self) -> None:
        logger.info("Running training on %d device(s).", self.gang.size)

        with self.profiler:
            while self.step_nr <= self.config.max_steps:
                with record_function(f"step_{self.step_nr}"):
                    if not self._train_step():
                        break

                if self._should_publish_train_metrics():
                    self._publish_train_metrics()

                if self._should_checkpoint():
                    self._checkpoint()

                if self._should_save_consolidated_model():
                    self._save_consolidated_model()

                if self._should_validate():
                    self._validate()

                self.profiler.step()

                self.step_nr += 1

                self.update_num_steps()

        logger.info("Finished training after %d step(s).", self.step_nr - 1)

    def restore(self) -> None:
        logger.info("Attempting to load last checkpoint.")

        step_nr, checkpoint = self.checkpoint_manager.load_last_checkpoint()

        logger.info("Checkpoint loaded, restoring training from step %d.", step_nr)

        self.load_state_dict(checkpoint)

        self.gang.barrier()

        logger.info("Training restored, resuming.")

        self.step_nr = step_nr + 1

        self.update_num_steps()

    def _train_step(self) -> bool:
        step_nr = self.step_nr

        step_stopwatch = Stopwatch(start=True, device=self.gang.device)

        stepped = False

        # We have to retry the step in case of a gradient overflow.
        while not stepped:
            # Collect batches.
            with record_function(f"step_{step_nr}_data_load"):
                batches = next(self.data_iter)

            logger.debug("Running training step %d.", step_nr)

            losses = []

            # Accumulate gradients.
            for batch_nr, batch in enumerate(batches):
                with record_function(f"step_{step_nr}_{batch_nr}_forward"):
                    with torch.autocast(device_type="cuda", dtype=self.dtype):
                        loss, extras = self._compute_loss(batch)

                with record_function(f"step_{step_nr}_{batch_nr}_backward"):
                    self.loss_scaler.backward(loss)

                losses.append(extras)

            # Update parameters.
            with record_function(f"step_{step_nr}_optimizer"):
                _, scale_result = self.loss_scaler.run_optimizer_step(step_nr)

            if scale_result.overflow:
                if scale_result.min_reached:
                    logger.error("Gradients scaled down to minimum at step %d. Stopping training.", step_nr)  # fmt: skip

                    raise FloatingPointError(
                        "The gradients are scaled down to minimum."
                    )

                logger.debug("Repeating training step %d.", step_nr)
            else:
                self.lr_scheduler.step()

                stepped = True

            # Reset.
            self.optimizer.zero_grad(set_to_none=True)

        with record_function(f"step_{step_nr}_metrics"):
            self.train_metric_bag.update_metrics(
                batches, losses, step_stopwatch.get_elapsed_time()
            )

        return stepped

    def _compute_loss(self, batch: UnitYBatch) -> Tensor:
        new_batch, text_targets, unit_targets = batch.as_input_and_target()

        output = cast(UnitYModelOutput, self.model(new_batch))

        return output.compute_loss(
            text_targets,
            unit_targets,
            text_loss_weight=self.config.text_loss_weight,
            aux_loss_type=self.config.aux_loss_type,
            aux_loss_weight=self.config.aux_loss_weight,
            ignore_text_prefix_size=self.config.ignore_text_prefix_size,
            ignore_unit_prefix_size=self.config.ignore_unit_prefix_size,
            label_smoothing=self.config.label_smoothing,
        )

    def _should_validate(self) -> bool:
        return self._should_do(self.config.validate_every_n_steps)

    @torch.inference_mode()
    def _validate(self) -> None:
        self.model.eval()

        logger.info("Starting validation after step %d.", self.step_nr)

        data_iter = self.dataset.read(
            "valid",
            self.text_tokenizer,
            self.unit_tokenizer,
            self.gang,
            self.dtype,
            self.config.max_seq_len,
            self.config.max_num_tokens,
            num_prefetch=self.config.num_prefetch,
            text_prefix_lang_tok=self.config.text_prefix_lang_tok,
            gcmvn_prosody_input=self.config.gcmvn_prosody_input,
            gcmvn_mean=self.gcmvn_mean,
            gcmvn_std=self.gcmvn_std,
        )

        for step_nr in count(start=1):
            step_stopwatch = Stopwatch(start=True, device=self.gang.device)

            try:
                batches = next(data_iter)
            except StopIteration:
                break

            logger.debug("Running validation step %d.", step_nr)

            losses = [self._compute_loss(batch)[1] for batch in batches]

            elapsed_time = step_stopwatch.get_elapsed_time()

            self.valid_metric_bag.update_metrics([losses], elapsed_time)

        self._publish_validation_metrics()

        logger.info("Validation complete, resuming training.")

        self.model.train()

    def update_num_steps(self) -> None:
        self.model.set_num_updates(self.step_nr)

    def _should_publish_train_metrics(self) -> bool:
        return self._should_do(self.config.publish_metrics_every_n_steps)

    def _publish_train_metrics(self) -> None:
        values = self.train_metric_bag.sync_and_compute_metrics()

        self.train_metric_bag.reset_batch_metrics()

        assert values is not None

        values["lr"] = get_effective_lr(self.lr_scheduler)

        if self.loss_scaler.enabled:
            values["grad_scale"] = self.loss_scaler.get_scale()

        values["wall_time"] = self.stopwatch.get_elapsed_time()

        record_metrics(self.metric_recorders, "Train", values, self.step_nr)

    def _publish_validation_metrics(self) -> None:
        values = self.valid_metric_bag.sync_and_compute_metrics()

        self.valid_metric_bag.reset_metrics()

        if self.gang.rank != 0:
            return

        assert values is not None

        values["wall_time"] = self.stopwatch.get_elapsed_time()

        record_metrics(self.metric_recorders, "Valid", values, self.step_nr)

    def _should_checkpoint(self) -> bool:
        return self._should_do(self.config.checkpoint_every_n_steps)

    def _should_save_consolidated_model(self) -> bool:
        return self.config.parallelism == "fsdp" and self._should_do(
            self.config.save_model_every_n_steps
        )

    def _checkpoint(self) -> None:
        logger.info("Saving checkpoint at step %d.", self.step_nr)

        checkpoint = self.state_dict()

        self.checkpoint_manager.save_checkpoint(
            self.step_nr, checkpoint, metadata={"config": self.config}
        )

        logger.info("Checkpoint saved.")

    def _save_consolidated_model(self) -> None:
        logger.info("Saving consolidated model at step %d.", self.step_nr)

        self.checkpoint_manager.save_consolidated_model(self.step_nr, self.model)

        logger.info("Consolidated model saved.")

    def _should_do(self, n_step: int) -> bool:
        return self.step_nr % n_step == 0


def load_unity_trainer(config: UnitYTrainConfig) -> UnitYTrainer:
    stopwatch = Stopwatch(start=True)

    setup_logging(
        log_file=config.output_dir.joinpath("logs/train_{rank}.log"), debug=False
    )

    setup_extensions()

    # In case we train on Ampere or later, use TF32.
    torch.set_float32_matmul_precision("high")

    logger.info(config)

    if config.parallelism == "none":
        gang = FakeGang()
    else:
        gang = ProcessGroupGang.init_default_process_group(ok_initialized=True)

    checkpoint_manager = FileCheckpointManager(
        config.output_dir.joinpath("checkpoints"),
        gang,
        distributed_fs=True,
    )

    rng_bag = RNGBag.from_device_defaults(CPU, gang.device)

    # Ensure that each run has deterministic behavior.
    rng_bag.manual_seed(config.seed)

    # load unit & text tokenizer for dataloader
    text_tokenizer = load_unity_text_tokenizer(config.tokenizer_name)
    unit_tokenizer = load_unity_unit_tokenizer(config.tokenizer_name)

    dtype = eval(config.dtype)

    dataset = load_unity_dataset(config.dataset_name)

    # Load the model.
    logger.info("Initializing model.")

    if isinstance(config.model_config_or_name, str):
        model_config = load_unity_config(config.model_config_or_name)
    else:
        model_config = config.model_config_or_name

    has_checkpoint = checkpoint_manager.has_checkpoint()

    broadcast_model = False

    if isinstance(config.model_config_or_name, str):
        # If we are finetuning and we don't have a checkpoint, load the
        # pretrained model on rank 0 and broadcast it to the gang.
        if not has_checkpoint:
            broadcast_model = True

        if not has_checkpoint and gang.rank == 0:
            init_device = gang.device
        else:
            init_device = META

        raw_model = load_unity_model(
            config.model_config_or_name,
            device=init_device,
            dtype=dtype,
        )
    else:
        raw_model = create_unity_model(model_config, device=gang.device, dtype=dtype)
        # below are fairsea2 checkpoints
        if config.load_fairseq1_s2t_weight:
            s2t_checkpoint = torch.load(config.load_fairseq1_s2t_weight)['model']
            raw_model.load_state_dict(s2t_checkpoint, strict=False)
        if config.load_prosody_encoder_weight:
            prosody_encoder_checkpoint = torch.load(config.load_prosody_encoder_weight)['model']
            raw_model.prosody_encoder_model.load_state_dict(prosody_encoder_checkpoint, strict=True)

    gang.barrier()

    model: Module
    if config.parallelism == "none":
        to_device(raw_model, gang.device)

        model = raw_model
    elif config.parallelism == "fsdp":
        wrap_policy, ignored_param_names = unity_wrap_policy()

        model = to_fsdp(
            raw_model,
            gang,
            wrap_policy,
            ignored_param_names=ignored_param_names,
            skip_init=has_checkpoint,
            broadcast_state=broadcast_model,
        )
        if gang.rank == 0:
            log_module(model, logger)

    elif config.parallelism == "ddp":
        model = DistributedDataParallel(
            raw_model,
            device_ids=[gang.device],
        )
        if gang.rank == 0:
            log_module(model, logger)

    trainer = UnitYTrainer(
        config,
        model,
        dataset,
        text_tokenizer,
        unit_tokenizer,
        gang,
        checkpoint_manager,
        rng_bag,
        stopwatch,
    )

    if has_checkpoint:
        trainer.restore()
    return trainer
