# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch

from seamless_communication.models.unit_extractor import UnitExtractor

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
        "--out_layer_number",
        type=int,
        help="Layer number of the feature extraction model to pull out features from.",
        default=35,
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
    logger.info(f"Converted to units: {units}")


if __name__ == "__main__":
    main()
