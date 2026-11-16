# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import json
from pathlib import Path

import pandas as pd
import pytest
from fairseq2.typing import Device

# The evaluation extras are not in install_requires.
pytest.importorskip("jiwer")
pytest.importorskip("sacrebleu")

from seamless_communication.cli.eval_utils.compute_metrics import (
    compute_quality_metrics,
)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize(
    "lang,reference,hypothesis,expected_score,metric",
    [
        ("eng", "we are", "we are here", 0.5, "wer"),
        ("eng", "we are here", "we are", 1 / 3, "wer"),
        ("cmn", "你好", "你好吗", 0.5, "cer"),
        ("cmn", "你好吗", "你好", 1 / 3, "cer"),
    ],
)
def test_asr_error_rate_uses_reference_length(
    tmp_path: Path,
    normalize: bool,
    lang: str,
    reference: str,
    hypothesis: str,
    expected_score: float,
    metric: str,
) -> None:
    manifest_path = tmp_path / "results.tsv"
    pd.DataFrame({"ref_tgt_text": [reference], "pred_tgt_text": [hypothesis]}).to_csv(
        manifest_path, sep="\t", index=False
    )

    filename = compute_quality_metrics(
        manifest_path,
        tmp_path,
        lang,
        "ASR",
        Device("cpu"),
        whisper_normalize_text_output=normalize,
    )

    result = json.loads((tmp_path / filename).read_text())
    assert result["score"] == pytest.approx(expected_score)
    assert result["signature"] == f"{metric} is {result['score']}"
