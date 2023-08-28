# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import torch
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
    parser.add_argument("input", type=str, help="Audio WAV file path.")
    parser.add_argument(
        "tgt_lang", type=str, help="Target language to translate/transcribe into."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language.",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the generated audio.",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Base model name (`seamlessM4T_medium`, `seamlessM4T_large`)",
        default="seamlessM4T_large",
    )

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
        logger.info(f"Running inference on the GPU in {dtype}.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.info(f"Running inference on the CPU in {dtype}.")

    asrbleu = ASRBleu(
        args.output_path,
    )
    asrbleu.compute_asr_bleu(
        args.input,
        args.tgt_lang,
        args.src_lang,
        args.model_name,
        device,
        dtype,
    )

if __name__ == "__main__":
    main()
