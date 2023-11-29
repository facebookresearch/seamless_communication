# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fire import Fire
import pandas as pd
import csv
from seamless_communication.cli.eval_utils.compute_metrics import (
    compute_quality_metrics,
)
import os
from fairseq2.typing import Device
from pathlib import Path


def create_output_manifest(
    generation_dir_path: str,
    generate_tsv_filename: str,
) -> pd.DataFrame:
    generate_df = pd.read_csv(
        f"{generation_dir_path}/{generate_tsv_filename}",
        sep="\t",
        quoting=csv.QUOTE_MINIMAL,
    )

    # fetch waveforms following indices from generate_df
    waveform_paths = []
    for idx in generate_df["id"]:
        waveform_path = f"{generation_dir_path}/waveform/{idx}_pred.wav"
        assert os.path.exists(waveform_path)
        waveform_paths.append(waveform_path)

    generate_df["hypo_audio"] = waveform_paths

    generate_df.set_index("id").to_csv(
        f"{generation_dir_path}/output_manifest.tsv",
        sep="\t",
        quoting=csv.QUOTE_MINIMAL,
    )
    return generate_df


def run_asr_bleu_expressive_model(
    generation_dir_path: str,
    generate_tsv_filename: str,
    tgt_lang: str,
) -> None:
    output_manifest_path = Path(generation_dir_path) / "output_manifest.tsv"

    if not output_manifest_path.exists():
        _ = create_output_manifest(
            generation_dir_path, generate_tsv_filename
        ).set_index("id")

    compute_quality_metrics(
        output_manifest_path,
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
