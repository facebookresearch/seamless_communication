# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path

import torch

from datasets import load_dataset
from seamless_communication.datasets.huggingface import (
    SpeechTokenizer,
)
from seamless_communication.models.unit_extractor import UnitExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("dataset")

UNITY_TO_COMMON_VOICE_LANG_MAPPING = {
    "eng": "en",
    "ita": "it",
    "afr": "af",
    "asm": "as",
    "bel": "be",
    "bul": "bg",
    "ben": "bn",
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "fin": "fi",
    "fra": "fr",
    "glg": "gl",
    "heb": "he",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "ind": "id",
    "ibo": "ig",
    "isl": "is",
    "jpn": "ja",
    "jav": "jv",
    "kaz": "kk",
    "kan": "kn",
    "kir": "ky",
    "kor": "ko",
    "lit": "lt",
    "mkd": "mk",
    "mlt": "mt",
    "mya": "my",
    "nld": "nl",
    "pan": "pa",
    "pol": "pl",
    "ron": "ro",
    "rus": "ru",
    "snd": "sd",
    "slk": "sk",
    "spa": "es",
    "srp": "sr",
    "swh": "sw",
    "tam": "ta",
    "tel": "te",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "uzn": "uz",
    "vie": "vi",
    "yor": "yo",
    "zul": "zu"
}

def _check_lang_code_mapping(lang: str) -> None:
    if lang not in UNITY_TO_COMMON_VOICE_LANG_MAPPING:
        raise ValueError(
            f"No language code mapping for {lang}(M4T)->??(CV). "
            "Please expand `UNITY_TO_COMMON_VOICE_LANG_MAPPING`"
        )

class UnitSpeechTokenizer(SpeechTokenizer):
    MODEL_NAME = "xlsr2_1b_v2"
    KMEANS_MODEL_URI = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    OUTPUT_LAYER_IDX = 34

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.unit_extractor = UnitExtractor(
            model_name_or_card=self.MODEL_NAME,
            kmeans_uri=self.KMEANS_MODEL_URI,
            device=self.device,
        )

    def encode(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return self.unit_extractor.predict(
            wav.to(self.device),
            out_layer_idx=self.OUTPUT_LAYER_IDX,
            sample_rate=sample_rate,
        )

def download_common_voice(lang: str, split: str, save_directory: str):
    _check_lang_code_mapping(lang)
    dataset = load_dataset('mozilla-foundation/common_voice_17_0', lang, split=split)
    manifest_path: str = os.path.join(save_directory, f"{split}_manifest.json")
    with open(manifest_path, "w") as fp_out:
        for idx, sample in enumerate(dataset, start=1):
            sample['lang'] = lang
            sample['waveform'] = None  # already extracted units
            fp_out.write(json.dumps(dataclasses.asdict(sample)) + "\n")
    logger.info(f"Saved {idx} samples for split={split} to {manifest_path}")
    logger.info(f"Manifest saved to: {manifest_path}")

def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Helper script to download training/evaluation dataset (Common Voice),"
            "extract units from target audio and save the dataset as a manifest "
            "consumable by `finetune.py`."
        )
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="Language of the dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split/shard to download (`train`, `validation`, `test`)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where the datasets will be stored with HuggingFace datasets cache files",
    )
    return parser

def main() -> None:
    args = init_parser().parse_args()
    download_common_voice(args.lang, args.split, args.save_dir)

if __name__ == "__main__":
    main()