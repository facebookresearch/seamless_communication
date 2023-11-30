# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import pandas as pd
import csv
import scipy
from typing import Dict


def get_pause(pause_data_tsv: str) -> Dict[str, float]:
    utt_pause_align_data = pd.read_csv(
        pause_data_tsv,
        sep="\t",
        quoting=csv.QUOTE_MINIMAL,
    )
    metrics = {}
    pause_duration_weight = (
        utt_pause_align_data.total_weight / utt_pause_align_data.total_weight.sum()
    )
    for score_name in [
        "wmean_duration_score",
        "wmean_alignment_score",
        "wmean_joint_score",
    ]:
        metrics[score_name] = (
            utt_pause_align_data[f"{score_name}"] * pause_duration_weight
        ).sum()
    return metrics


def get_rate(target_speech_tsv: str, source_speech_tsv: str) -> float:
    speech_unit = "syllable"

    target_speech_df = pd.read_csv(
        target_speech_tsv, sep="\t", quoting=csv.QUOTE_MINIMAL
    ).set_index("id")
    source_speech_df = pd.read_csv(
        source_speech_tsv, sep="\t", quoting=csv.QUOTE_MINIMAL
    ).set_index("id")

    # using "syllable" speech unit for rate computation
    src_speech_rate = source_speech_df[f"speech_rate_{speech_unit}"].to_numpy()
    tgt_speech_rate = target_speech_df[f"speech_rate_{speech_unit}"].to_numpy()
    src_tgt_spearman = scipy.stats.spearmanr(src_speech_rate, tgt_speech_rate)
    return src_tgt_spearman.correlation  # type: ignore[no-any-return]
