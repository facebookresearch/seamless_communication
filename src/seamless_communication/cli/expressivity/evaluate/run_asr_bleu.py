# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from fire import Fire
import pandas as pd
import csv
from seamless_communication.cli.eval_utils.compute_metrics import (
    compute_quality_metrics,
)
import os
from fairseq2.typing import Device
from pathlib import Path


def run_asr_bleu_expressive_model(
    generation_dir_path: str,
    generate_tsv_filename: str,
    tgt_lang: str,
):
    compute_quality_metrics(
        f"{generation_dir_path}/{generate_tsv_filename}",
        Path(generation_dir_path),
        tgt_lang,
        "S2ST",
        device=Device("cuda"),
        ref_text_col_name="tgt_text",
        pred_text_col_name="s2t_out",
        pred_audio_col_name="hypo_audio",
    )


if __name__ == "__main__":
    Fire(run_asr_bleu_expressive_model)
