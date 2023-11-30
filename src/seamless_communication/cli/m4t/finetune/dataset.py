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

from seamless_communication.datasets.huggingface import (
    Speech2SpeechFleursDatasetBuilder,
    SpeechTokenizer,
)
from seamless_communication.models.unit_extractor import UnitExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("dataset")


# Full list of FLEURS langcodes is available at https://huggingface.co/datasets/google/fleurs
# Full list of M4T langcodes is available
# in paper "SeamlessM4T—Massively Multilingual & Multimodal Machine Translation" (Table 5)
UNITY_TO_FLEURS_LANG_MAPPING = {
    "eng": "en_us",
    "ita": "it_it",
    "afr": "af_za",
    "asm": "as_in",
    "bel": "be_by",
    "bul": "bg_bg",
    "ben": "bn_in",
    "cat": "ca_es",
    "ces": "cs_cz",
    "dan": "da_dk",
    "deu": "de_de",
    "ell": "el_gr",
    "fin": "fi_fi",
    "fra": "fr_fr",
    "glg": "gl_es",
    "heb": "he_il",
    "hin": "hi_in",
    "hrv": "hr_hr",
    "hun": "hu_hu",
    "ind": "id_id",
    "ibo": "ig_ng",
    "isl": "is_is",
    "ita": "it_it",
    "jpn": "ja_jp",
    "jav": "jv_id",
    "kaz": "kk_kz",
    "kan": "kn_in",
    "kir": "ky_kg",
    "kor": "ko_kr",
    "lit": "lt_lt",
    "mkd": "mk_mk",
    "mlt": "mt_mt",
    "mya": "my_mm",
    "nld": "nl_nl",
    "pan": "pa_in",
    "pol": "pl_pl",
    "ron": "ro_ro",
    "rus": "ru_ru",
    "snd": "sd_in",
    "slk": "sk_sk",
    "srp": "sr_rs",
    "swh": "sw_ke",
    "tam": "ta_in",
    "tel": "te_in",
    "tha": "th_th",
    "tur": "tr_tr",
    "ukr": "uk_ua",
    "urd": "ur_pk",
    "uzn": "uz_uz",
    "vie": "vi_vn",
    "yor": "yo_ng",
    "zul": "zu_za",
}


def _check_lang_code_mapping(lang: str) -> None:
    if lang not in UNITY_TO_FLEURS_LANG_MAPPING:
        raise ValueError(
            f"No language code mapping for {lang}(M4T)->??(FLEURs). "
            "Please expand `UNITY_TO_FLEURS_LANG_MAPPING`"
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


def download_fleurs_dataset(
    source_lang: str,
    target_lang: str,
    split: str,
    save_directory: str,
) -> str:
    _check_lang_code_mapping(source_lang)
    _check_lang_code_mapping(target_lang)
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )
    tokenizer = UnitSpeechTokenizer(device=device)
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
        for idx, sample in enumerate(dataset_iterator.__iter__(), start=1):
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
        help="Dataset split/shard to download (`train`, `validation`, `test`)",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="Directory where the datastets will be stored with HuggingFace datasets cache files",
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()
    manifest_path = download_fleurs_dataset(
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        split=args.split,
        save_directory=args.save_dir,
    )
    logger.info(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
