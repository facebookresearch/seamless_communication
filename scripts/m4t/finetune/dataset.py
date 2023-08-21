# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import dataclasses
import json
import logging
import os
from argparse import Namespace
from pathlib import Path

from stopes.hub import load_config
from stopes.speech.tokenizers import SpeechTokenizer, SpeechTokenizerConfig

from seamless_communication.datasets.huggingface import (
    Speech2SpeechFleursDatasetBuilder,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("dataset")


# List of FLEURS langcodes is available at https://huggingface.co/datasets/google/fleurs
# List of M4T langcodes is available in yaml: src/seamless_communication/assets/cards/unity_nllb-100.yaml
UNITY_TO_FLEURS_LANG_MAPPING = {
    "eng": "en_us",
    "ita": "it_it",
    "kor": "ko_kr",
}


def download_fleurs_dataset(
    source_lang: str,
    target_lang: str,
    split: str,
    unit_extractor_config: str,
    save_directory: str,
) -> str:
    tokenizer_conf: SpeechTokenizerConfig = load_config(
        unit_extractor_config, namespace=""
    )
    tokenizer: SpeechTokenizer = SpeechTokenizer.build(tokenizer_conf)
    dataset_iterator = Speech2SpeechFleursDatasetBuilder(
        source_lang=UNITY_TO_FLEURS_LANG_MAPPING[source_lang],
        target_lang=UNITY_TO_FLEURS_LANG_MAPPING[target_lang],
        dataset_cache_dir=save_directory,
        speech_tokenizer=tokenizer,
        skip_source_audio=True,  # don't extract units from source audio
        skip_target_audio=False,
        split=split,
    )
    manifest_path: str = os.path.join(save_directory, f"{split}_manifest.json")
    with open(manifest_path, "w") as fp_out:
        for idx, sample in enumerate(dataset_iterator, start=1):
            # correction as FleursDatasetBuilder return fleurs lang codes
            sample.source.lang = source_lang
            sample.target.lang = target_lang
            sample.target.waveform = None  # already extracted units
            fp_out.write(json.dumps(dataclasses.asdict(sample)) + "\n")
    logger.info(f"Saved {idx} samples for split={split} to {manifest_path}")
    return manifest_path


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Helper script to download training/evaluation dataset (FLEURS),"
            "extract units from target audio and save the dataset as a manifest "
            "consumable by `finetune.py`."
        )
    )
    parser.add_argument(
        "--source_lang",
        type=str,
        required=True,
        help="M4T langcode of the dataset SOURCE language",
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        required=True,
        help="M4T langcode of the dataset TARGET language",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split/shard to download (`train`, `test`)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where the datastets will be stored with HuggingFace datasets cache files",
    )
    return parser


def main(args: Namespace) -> None:
    manifest_path = download_fleurs_dataset(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        # TODO: remove hardcoded path
        unit_extractor_config="/checkpoint/krs/unit_extraction/xlsr1b/lang41_10k_xlsr_lyr35.yaml",
        split=args.split,
        save_directory=args.save_dir,
    )
    logger.info(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    args = init_parser().parse_args()
    main(args)
