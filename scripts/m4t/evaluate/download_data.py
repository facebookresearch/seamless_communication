# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import scipy.io.wavfile as wavfile
import torch
from seamless_communication.datasets.huggingface import \
    Speech2SpeechFleursDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

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


def make_directories(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def combine_texts(texts, output_path):
    with open(output_path, "w") as output_file:
        for text in texts:
            output_file.write(text + "\n")


def download_datasets(language_pairs, split, num_datasets, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for source_lang, target_lang in language_pairs:
        source_fleurs_code = UNITY_TO_FLEURS_LANG_MAPPING[source_lang]
        target_fleurs_code = UNITY_TO_FLEURS_LANG_MAPPING[target_lang]
        lang_pair = f"{source_lang}-{target_lang}"

        lang_output_dir = os.path.join(output_directory, lang_pair)
        lang_source_audio_dir = os.path.join(
            lang_output_dir, f"source_audio_{source_lang}"
        )
        lang_target_texts_dir = os.path.join(
            lang_output_dir, f"target_text_{target_lang}"
        )

        make_directories(
            lang_output_dir,
            lang_source_audio_dir,
            lang_target_texts_dir,
        )

        logger.info(f"Downloading datasets for language pair: {lang_pair}")

        cache_dir = os.path.join(output_directory, "cache")

        dataset_builder = Speech2SpeechFleursDatasetBuilder(
            source_lang=source_fleurs_code,
            target_lang=target_fleurs_code,
            split=split,
            skip_source_audio=False,
            skip_target_audio=False,
            audio_dtype=torch.float32,
            dataset_cache_dir=cache_dir,
            speech_tokenizer=None,
        )

        dataset_count = 0
        target_texts = []

        for _, lang_pair_sample in enumerate(dataset_builder):

            source_sample = lang_pair_sample.source
            target_sample = lang_pair_sample.target

            source_audio = source_sample.waveform.numpy()
            target_text = target_sample.text

            source_audio_path = os.path.join(
                lang_source_audio_dir, f"{dataset_count}_pred.wav"
            )

            # Save audio data as WAV files
            wavfile.write(source_audio_path, source_sample.sampling_rate, source_audio)

            target_texts.append(target_text)

            logger.info(
                f"Dataset {dataset_count} - Source Audio Path: {source_audio_path}"
            )
            logger.info(
                f"Dataset {dataset_count} - Source Language: {source_sample.lang}"
            )
            logger.info(
                f"Dataset {dataset_count} - Target Language: {target_sample.lang}"
            )
            logger.info(f"Dataset {dataset_count} - Target Text: {target_text}")
            print("=" * 100)

            dataset_count += 1
            if dataset_count >= num_datasets:
                break

        combine_texts(
            target_texts,
            os.path.join(lang_target_texts_dir, "target_text_references.txt"),
        )

        logger.info(
            f"Downloaded and saved {dataset_count} datasets for language pair: {lang_pair}"
        )
        print("=" * 100)
