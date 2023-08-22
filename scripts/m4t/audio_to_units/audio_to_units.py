# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import torch
import torchaudio
from seamless_communication.models.unit_extraction import UnitExtractor
from seamless_communication.models.inference import Translator
from seamless_communication.models.vocoder import load_vocoder_model, Vocoder
from itertools import groupby


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw audio to units (and optionally audio) using UnitExtractor."
    )
    parser.add_argument("audio", type=str, help="Audio WAV file path.")
    parser.add_argument(
        "--kmeans_uri",
        type=str,
        help="URL path to the K-Means model.",
        default="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Feature extraction model name (`xlsr2_1b_v2`)",
        default="xlsr2_1b_v2",
    )
    parser.add_argument(
        "--vocoder_name", type=str, help="Vocoder name", default="vocoder_36langs"
    )
    parser.add_argument(
        "--out_layer_number",
        type=int,
        help="Layer number of the feature extraction model to pull out features from.",
        default=35,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated audio.",
        default=None,
    )
    parser.add_argument(
        "--src_lang", type=str, help="Source language of the audio.", default=None
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Running unit_extraction on the GPU.")
    else:
        device = torch.device("cpu")
        logger.info("Running unit_extraction on the CPU.")

    unit_extractor = UnitExtractor(args.model_name, args.kmeans_uri, device=device)
    units = unit_extractor.predict(args.audio, args.out_layer_number - 1)

    if args.output_path is not None:

        if args.src_lang is None:
            raise ValueError("src_lang must be provided to resynthesize the audio.")

        def reduce_list(lst):
            return [key for key, _ in groupby(lst)]

        reduced_units = reduce_list(units.cpu().tolist())

        vocoder: Vocoder = Translator.load_model_for_inference(
            load_vocoder_model, args.vocoder_name, device, torch.float32
        )
        wav = vocoder(reduced_units, args.src_lang, spkr=-1, dur_prediction=True)

        torchaudio.save(
            args.output_path,
            wav[0].cpu(),
            sample_rate=16000,
        )


if __name__ == "__main__":
    main()
