# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import logging
from m4t_scripts.finetune.dataset import download_fleurs_dataset, UNITY_TO_FLEURS_LANG_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("dataset")

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
        # Get the language mapping from download_fleurs_dataset
        source_mapping = UNITY_TO_FLEURS_LANG_MAPPING[source_lang]
        target_mapping = UNITY_TO_FLEURS_LANG_MAPPING[target_lang]
        
        lang_dir = f"{source_lang}-{target_lang}"
        lang_pair = f"{source_mapping}-{target_mapping}"

        lang_output_dir = os.path.join(output_directory, lang_dir)
        lang_source_audio_dir = os.path.join(
            lang_output_dir, f"source_audio_{source_lang}"
        )
        lang_target_texts_dir = os.path.join(
            lang_output_dir, f"target_texts_{target_lang}"
        )

        make_directories(
            lang_output_dir,
            lang_source_audio_dir,
            lang_target_texts_dir,
        )

        logger.info(f"Downloading datasets for language pair: {lang_pair}")

        # Download the dataset using the download_fleurs_dataset function
        manifest_path = download_fleurs_dataset(
            source_lang=source_lang,
            target_lang=target_lang,
            split=split,
            save_directory=output_directory,
        )

        # Read the manifest file and parse JSON objects
        with open(manifest_path, "r") as manifest_file:
            manifest_entries = [json.loads(line.strip()) for line in manifest_file]

        dataset_count = 0
        target_texts = []

        for entry in manifest_entries:
            source_audio_path = os.path.join(
                lang_source_audio_dir, f"{entry['source']['id']}_pred.wav"
            )

            # Copy the source audio to the output directory
            source_audio_source_path = entry["source"]["audio_local_path"]
            shutil.copy(source_audio_source_path, source_audio_path)

            target_text = entry["target"]["text"]
            target_texts.append(target_text)

            logger.info(f"Dataset {dataset_count} - Source Audio Path: {source_audio_path}")
            logger.info(f"Dataset {dataset_count} - Source Language: {source_lang}")
            logger.info(f"Dataset {dataset_count} - Target Language: {target_lang}")
            logger.info(f"Dataset {dataset_count} - Target Text: {target_text}")
            print("=" * 100)

            dataset_count += 1
            if dataset_count >= num_datasets:
                break

        # Combine target texts and save to a single file
        combine_texts(
            target_texts,
            os.path.join(lang_target_texts_dir, "references.txt"),
        )

        logger.info(
            f"Downloaded and saved {dataset_count} datasets for language pair: {lang_pair}"
        )
        print("=" * 100)
