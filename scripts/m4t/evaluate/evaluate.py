# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from m4t_scripts.evaluate.asr_bleu import ASRBleu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="M4T inference on supported tasks using Translator."
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to save results.",
    )
    parser.add_argument(
        "lang_dir", type=str, help="Language direction for translation."
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Test/train/validation split.",
        default="test",
    )
    parser.add_argument(
        "--num_data_pairs",
        type=int,
        help="Number of audio/text pairs to evaluate",
        default=5,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
        default="seamlessM4T_large",
    )
    parser.add_argument(
        "--eval_first_pass",
        type=bool,
        help="Save first pass text data and BLEU score.",
        default=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Name of passed dataset.",
        default="fleurs",
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        help="Format of audio file (eg. n_pred.wav).",
        default="n_pred.wav",
    )

    args = parser.parse_args()

    asrbleu = ASRBleu(
        args.output_dir,
    )
    asrbleu.compute_asr_bleu(
        args.lang_dir,
        args.split,
        args.num_data_pairs,
        args.model_name,
        args.eval_first_pass,
        args.dataset,
        args.audio_format,
    )


if __name__ == "__main__":
    main()
