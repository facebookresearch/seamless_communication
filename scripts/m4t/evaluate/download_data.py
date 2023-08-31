# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
import scipy.io.wavfile as wavfile
from seamless_communication.datasets.huggingface import Speech2SpeechFleursDatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

def make_directories(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)

def combine_texts(texts, output_path):
    with open(output_path, 'w') as output_file:
        for text in texts:
            output_file.write(text + '\n')

def download_datasets(languages, num_datasets_per_language, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for lang in languages:
        lang_output_dir = os.path.join(output_directory, lang)
        lang_source_audio_dir = os.path.join(lang_output_dir, 'source_audio')
        lang_target_text_dir = os.path.join(lang_output_dir, 'target_text')
        lang_source_text_dir = os.path.join(lang_output_dir, 'source_text')
        lang_target_audio_dir = os.path.join(lang_output_dir, 'target_audio')

        make_directories(
            lang_output_dir,
            lang_source_audio_dir,
            lang_target_text_dir,
            lang_source_text_dir,
            lang_target_audio_dir
        )

        logger.info(f"Downloading datasets for language: {lang}")

        dataset_builder = Speech2SpeechFleursDatasetBuilder(
            source_lang=lang,
            target_lang=lang,
            split="test",
            skip_source_audio=False,
            skip_target_audio=False,
            audio_dtype=torch.float32,
            dataset_cache_dir=None,
            speech_tokenizer=None,
        )

        dataset_count = 0
        source_texts = []  
        target_texts = []  

        for lang_pair_sample in dataset_builder:
            source_sample = lang_pair_sample.source
            target_sample = lang_pair_sample.target

            source_audio = source_sample.waveform.numpy()
            target_audio = target_sample.waveform.numpy()

            source_text = source_sample.text
            target_text = target_sample.text

            source_audio_path = os.path.join(lang_source_audio_dir, f"source_{dataset_count}.wav")
            target_audio_path = os.path.join(lang_target_audio_dir, f"target_{dataset_count}.wav")

            # Save audio data as WAV files
            wavfile.write(source_audio_path, source_sample.sampling_rate, source_audio)
            wavfile.write(target_audio_path, target_sample.sampling_rate, target_audio)

            source_texts.append(source_text)
            target_texts.append(target_text)

            logger.info(f"Dataset {dataset_count} - Source Audio Path: {source_audio_path}")
            logger.info(f"Dataset {dataset_count} - Source Language: {source_sample.lang}")
            logger.info(f"Dataset {dataset_count} - Target Language: {target_sample.lang}")
            logger.info(f"Dataset {dataset_count} - Target Text: {target_text}")
            print("=" * 100)

            dataset_count += 1
            if dataset_count >= num_datasets_per_language:
                break

        
        combine_texts(source_texts, os.path.join(lang_source_text_dir, 'source_references.txt'))
        combine_texts(target_texts, os.path.join(lang_target_text_dir, 'target_references.txt'))

        logger.info(f"Downloaded and saved {dataset_count} datasets for language: {lang}")
        print("=" * 100)

languages = ["hi_in", "af_za"]
num_datasets_per_language = 10
output_directory = "./downloaded_data"

download_datasets(languages, num_datasets_per_language, output_directory)
