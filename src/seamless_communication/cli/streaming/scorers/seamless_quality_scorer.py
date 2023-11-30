# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional

import pandas
from fairseq2.typing import Device
from seamless_communication.cli.eval_utils import compute_quality_metrics
from simuleval.evaluator.instance import LogInstance
from simuleval.evaluator.scorers.quality_scorer import (
    QualityScorer,
    register_quality_scorer,
)


@register_quality_scorer("SEAMLESS_QUALITY_SCORER")
class SeamlessQualityScorer(QualityScorer):  # type: ignore
    def __init__(
        self,
        tgt_lang: str,
        task: str,
        output_dir: str,
        device: Device = "cuda:0",
        whisper_model_name: str = "large",
        whisper_normalize_text_output: Optional[bool] = None,
        ref_text_col_name: str = "ref_tgt_text",
        pred_text_col_name: str = "pred_tgt_text",
        pred_audio_col_name: str = "pred_tgt_audio",
    ) -> None:
        super().__init__()
        self.tgt_lang = tgt_lang
        self.task = task.upper()
        self.device = device
        self.output_dir = Path(output_dir)
        self.whisper_model_name = whisper_model_name
        self.whisper_normalize_text_output = whisper_normalize_text_output
        if self.whisper_normalize_text_output is None:
            self.whisper_normalize_text_output = (
                False if self.task in ["S2TT", "S2ST", "T2TT"] else True
            )
        self.ref_text_col_name = ref_text_col_name
        self.pred_text_col_name = pred_text_col_name
        self.pred_audio_col_name = pred_audio_col_name

    def __call__(self, instances: Dict[int, LogInstance]) -> float:
        references = [ins.reference for ins in instances.values()]
        df = pandas.DataFrame({self.ref_text_col_name: references})
        if self.task in ["ASR", "S2TT", "T2TT"]:
            predictions = [ins.prediction for ins in instances.values()]
            df[self.pred_text_col_name] = predictions
        else:
            predictions = [ins.prediction for ins in instances.values()]
            df[self.pred_audio_col_name] = predictions

        df.to_csv(
            self.output_dir / "results.tsv",
            sep="\t",
            quoting=3,
            encoding="utf-8",
        )
        filename = compute_quality_metrics(
            self.output_dir / "results.tsv",
            self.output_dir,
            self.tgt_lang,
            self.task,
            self.device,
            self.whisper_model_name,
            self.whisper_normalize_text_output,
            self.ref_text_col_name,
            self.pred_text_col_name if self.task in ["ASR", "S2TT", "T2TT"] else None,
            self.pred_audio_col_name,
        )

        with open(self.output_dir / filename, "r") as f:
            corpus_metric_score = json.load(f)["score"]

        return corpus_metric_score  # type: ignore[no-any-return]

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument("--task", type=str, help="Task to evaluate", required=True)
        parser.add_argument(
            "--tgt-lang",
            type=str,
            help="Target language to translate/transcribe into.",
            required=True,
        )
        parser.add_argument(
            "--whisper-model-name", type=str, help="Whisper model name", default="large"
        )
        parser.add_argument(
            "--whisper-normalize-text-output",
            action="store_true",
            help="Normalize text output",
            default=None,
        )
        parser.add_argument(
            "--ref-text-col-name",
            type=str,
            help="Reference text column name",
            default="ref_tgt_text",
        )
        parser.add_argument(
            "--pred-text-col-name",
            type=str,
            help="Prediction text column name",
            default="pred_tgt_text",
        )
        parser.add_argument(
            "--pred-audio-col-name",
            type=str,
            help="Prediction audio column name",
            default="pred_tgt_audio",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> SeamlessQualityScorer:
        return cls(
            tgt_lang=args.tgt_lang,
            task=args.task,
            output_dir=args.output,
            device=getattr(args, "device", "cpu"),
            whisper_model_name=args.whisper_model_name,
            whisper_normalize_text_output=args.whisper_normalize_text_output,
            ref_text_col_name=args.ref_text_col_name,
            pred_text_col_name=args.pred_text_col_name,
            pred_audio_col_name=args.pred_audio_col_name,
        )
