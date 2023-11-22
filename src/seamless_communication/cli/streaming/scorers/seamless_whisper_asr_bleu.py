# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Dict, List

from sacrebleu.metrics.bleu import BLEU
from seamless_communication.cli.eval_utils import get_tokenizer, LANG3_LANG2
from simuleval.evaluator.instance import LogInstance
from simuleval.evaluator.scorers.quality_scorer import (
    WhisperASRSacreBLEUScorer,
    register_quality_scorer,
)
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer


def normalize_text_whisper(sentences: List[str], lang: str) -> List[str]:
    if lang in ["en", "eng"]:
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()
    normalized_sentences = []
    for text in sentences:
        normalized_sentences.append(normalizer(text))
    return normalized_sentences


@register_quality_scorer("SEAMLESS_WHISPER_ASR_BLEU")
class SeamlessWhisperASRSacreBLEUScorer(WhisperASRSacreBLEUScorer):  # type: ignore
    def __init__(
        self,
        tokenizer: str = "13a",
        target_lang: str = "en",
        model_size: str = "base",
        lowercase: bool = False,
        remove_punctuations: bool = False,
        normalize_asr_bleu_references: bool = False,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.target_lang = target_lang
        self.model_size = model_size
        self.lowercase = lowercase
        self.remove_punctuations = remove_punctuations
        self.normalize_asr_bleu_references = normalize_asr_bleu_references

    def __call__(self, instances: Dict[int, LogInstance]) -> float:
        transcripts = self.asr_transcribe(instances)
        references = [[ins.reference for ins in instances.values()]]

        if self.normalize_asr_bleu_references:
            transcripts = normalize_text_whisper(transcripts, self.target_lang)
            references = [normalize_text_whisper(references[0], self.target_lang)]

        score = (
            BLEU(tokenize=self.tokenizer).corpus_score(transcripts, references).score
        )
        return score  # type: ignore[no-any-return]

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        WhisperASRSacreBLEUScorer.add_args(parser)
        parser.add_argument(
            "--normalize-asr-bleu-references",
            action="store_true",
            help="Normalize asr transcript and reference",
        )

    @classmethod
    def from_args(cls, args: Namespace) -> SeamlessWhisperASRSacreBLEUScorer:
        sacrebleu_tokenizer = get_tokenizer(args.tgt_lang)
        tgt_lang_2ltr = LANG3_LANG2[args.tgt_lang]
        return cls(
            sacrebleu_tokenizer,
            tgt_lang_2ltr,
            args.whisper_model_size,
            args.transcript_lowercase,
            args.transcript_non_punctuation,
            args.normalize_asr_bleu_references,
        )
