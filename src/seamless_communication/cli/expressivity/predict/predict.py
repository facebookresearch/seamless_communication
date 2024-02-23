# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path

import torch
import torchaudio

from seamless_communication.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)
from seamless_communication.inference import ExpressiveTranslator
from seamless_communication.store import add_gated_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Running SeamlessExpressive inference."
    )
    parser.add_argument("input", type=Path, help="Audio WAV file path.")

    parser = add_inference_arguments(parser)
    parser.add_argument(
        "--gated-model-dir",
        type=Path,
        required=False,
        help="SeamlessExpressive model directory.",
    )
    parser.add_argument(
        "--duration_factor",
        type=float,
        help="The duration factor for NAR T2U model.",
        default=1.0,
    )
    args = parser.parse_args()

    if not args.tgt_lang or args.output_path is None:
        raise Exception(
            "--tgt_lang, --output_path must be provided for SeamlessExpressive inference."
        )

    if args.gated_model_dir:
        add_gated_assets(args.gated_model_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running inference on {device=} with {dtype=}.")

    expressive_translator = ExpressiveTranslator(
        args.model_name, args.vocoder_name, device, dtype
    )

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    text_output, speech_output = expressive_translator.predict(
        args.input,
        args.tgt_lang,
        text_generation_opts,
        unit_generation_opts,
        args.unit_generation_ngram_filtering,
        args.duration_factor,
    )

    logger.info(f"Saving expressive translated audio in {args.tgt_lang}")
    torchaudio.save(
        args.output_path,
        speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
        sample_rate=speech_output.sample_rate,
    )

    logger.info(f"Translated text in {args.tgt_lang}: {text_output[0]}")


if __name__ == "__main__":
    main()
