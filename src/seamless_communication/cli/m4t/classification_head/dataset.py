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
import torchaudio

from datasets import load_dataset
from seamless_communication.datasets.huggingface import (
    SpeechTokenizer,
)
from seamless_communication.models.unit_extractor import UnitExtractor

from seamless_communication.datasets.datatypes import LangPairSample, MultimodalSample

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
    "zul": "zu",
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


def download_common_voice(
    lang: str, split: str, save_directory: str, max_samples: int
) -> None:
    _check_lang_code_mapping(lang)
    mozilla_lang = UNITY_TO_COMMON_VOICE_LANG_MAPPING[lang]
    dataset = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        mozilla_lang,
        split=split,
        token=os.environ.get("HF_TOKEN"),
        streaming=True,
    )
    audio_dir = os.path.join(save_directory, "audio")
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    manifest_path: str = os.path.join(save_directory, f"{split}_{lang}_manifest.json")
    with open(manifest_path, "w") as fp_out:
        for idx, sample in enumerate(dataset, start=1):
            wav = torch.from_numpy(sample["audio"]["array"]).unsqueeze(0)
            logger.info(f"WAV SHAPE {wav.shape}")
            sampling_rate = sample["audio"]["sampling_rate"]
            audio_path = (
                split
                + "_"
                + os.path.basename(sample["audio"]["path"]).split(".")[0]
                + ".wav"
            )
            audio_path = os.path.join(audio_dir, audio_path)
            target_sr = 16000
            wav = torchaudio.functional.resample(
                wav, orig_freq=sampling_rate, new_freq=target_sr
            )
            torchaudio.save(audio_path, wav, target_sr)
            sample = MultimodalSample(
                id=idx, lang=lang, text=sample["sentence"], audio_local_path=audio_path
            )
            sample = LangPairSample(sample, sample)
            fp_out.write(json.dumps(dataclasses.asdict(sample)) + "\n")
            fp_out.flush()
            if idx == max_samples:
                break
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max samples to fetch",
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()
    download_common_voice(
        lang=args.lang,
        split=args.split,
        save_directory=args.save_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
