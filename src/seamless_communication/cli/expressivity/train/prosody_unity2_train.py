# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
from pathlib import Path
import torch

from fairseq2 import setup_extensions
from seamless_communication.train.unity2.trainer import UnitYTrainConfig, load_unity_trainer
from seamless_communication.models.unity import unity_archs
from seamless_next.cluster import Cluster, ClusterConfig


def train_unity(config: UnitYTrainConfig) -> None:
    setup_extensions()
    trainer = load_unity_trainer(config)
    trainer.run()


if __name__ == "__main__":
    model_config = unity_archs.get_config("expressivity_v2_trainable")

    train_config = UnitYTrainConfig(
        model_config_or_name=model_config,
        tokenizer_name="seamless_expressivity",
        dataset_name="seamless_expressive_eng_cmn",
        gcmvn_prosody_input=True,
        gcmvn_stats_name="vocoder_pretssel",
        output_dir=Path("training_logs/"),
        dtype="torch.bfloat16",
        parallelism="fsdp",
        num_prefetch=4,
        shuffle_window_size=0,
        publish_metrics_every_n_steps=1,
        gradient_accumulation=4,
        ignore_text_prefix_size=1,
        aux_loss_type="ctc",
        aux_loss_weight=1.6,
        label_smoothing=0.2,
        load_fairseq1_s2t_weight="/fsx-ust/shared/seamless23_oss/m4t_v2_s2t.pt",
        load_prosody_encoder_weight="/fsx-ust/shared/seamless23_oss/prosody_encoder_pretssel_pretrained.pt",
    )

    cluster_config = ClusterConfig(
        cluster="local",
        parallelism=1,
        partition="",
        num_nodes=1,
        num_gpus_per_node=8,
        cpus_per_task=10,
        log_dir=train_config.output_dir.joinpath("submitit"),
        timeout=timedelta(minutes=1000),
    )

    cluster = Cluster(cluster_config)

    cluster.run_job(train_unity, train_config)
    # train_unity(train_config)
