# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
from pathlib import Path

import torch
from seamless_communication.train.unity2.trainer import UnitYTrainConfig, load_unity_trainer
from seamless_communication.models.unity import unity_archs
# from seamless_next.cluster import Cluster, ClusterConfig


# def main() -> None:

#     user = os.getenv("USER")

#     output_dir = Path(f"/checkpoint/{user}/nllb100-example")

#     cluster_config = ClusterConfig(
#         cluster="local",
#         parallelism=1,
#         partition="",
#         num_nodes=1,
#         num_gpus_per_node=1,
#         cpus_per_task=1,
#         log_dir=output_dir.joinpath("submitit"),
#         timeout=timedelta(minutes=1000),
#     )

#     cluster = Cluster(cluster_config)

#     # Load Jean's new 128k NLLB tokenizer.
#     tokenizer = load_nllb_tokenizer("nllb-100-128k")

#     model_config = nllb_archs.get_config("dense_1b")

#     # We adapt the existing NLLB-200 dense 1B to 100 languages.
#     model_config.vocab_info = tokenizer.vocab_info

#     train_config = NllbTrainConfig(
#         model_config_or_name=model_config,
#         tokenizer_name="nllb-100-128k",
#         dataset_name="nllb-100-dataset",
#         output_dir=output_dir,
#         parallelism="fsdp",
#         dtype=torch.float16,
#         debug=True,
#     )

#     cluster.run_job(train_nllb, train_config)


if __name__ == "__main__":
    model_config = unity_archs.get_config("expressivity_v2_trainable")

    train_config = UnitYTrainConfig(
        model_config_or_name=model_config,
        tokenizer_name="seamless_expressivity",
        dataset_name="seamless_expressive_eng_cmn",
        gcmvn_prosody_input=True,
        gcmvn_stats_name="vocoder_pretssel",
        output_dir=Path("~/tmp/test"),
        dtype="torch.float16",
        parallelism="none",
        num_prefetch=3,
        shuffle_window_size=0,
        use_submitit=False,
        gradient_accumulation=1,
        ignore_text_prefix_size=1,
        aux_loss_type="ctc",
        aux_loss_weight=1.6,
        label_smoothing=0.2,
        load_fairseq1_s2t_weight="/fsx-ust/shared/seamless23_oss/m4t_v2_s2t.pt",
        load_prosody_encoder_weight="/fsx-ust/shared/seamless23_oss/prosody_encoder_pretssel_pretrained.pt",
    )

    trainer = load_unity_trainer(train_config)

    trainer.run()
