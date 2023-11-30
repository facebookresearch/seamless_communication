# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import torch
import torchaudio
from fairseq2.generation import NGramRepeatBlockProcessor

from seamless_communication.inference import SequenceGeneratorOptions, Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def add_inference_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--task", type=str, help="Task type")
    parser.add_argument(
        "--tgt_lang", type=str, help="Target language to translate/transcribe into."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language, only required if input is text.",
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to save the generated audio.",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Base model name (`seamlessM4T_medium`, "
            "`seamlessM4T_large`, `seamlessM4T_v2_large`)"
        ),
        default="seamlessM4T_v2_large",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        help="Vocoder model name",
        default="vocoder_v2",
    )
    # Text generation args.
    parser.add_argument(
        "--text_generation_beam_size",
        type=int,
        help="Beam size for incremental text decoding.",
        default=5,
    )
    parser.add_argument(
        "--text_generation_max_len_a",
        type=int,
        help="`a` in `ax + b` for incremental text decoding.",
        default=1,
    )
    parser.add_argument(
        "--text_generation_max_len_b",
        type=int,
        help="`b` in `ax + b` for incremental text decoding.",
        default=200,
    )
    parser.add_argument(
        "--text_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental text decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        help="Size of ngram repeat block for both text & unit decoding.",
        default=4,
    )
    # Unit generation args.
    parser.add_argument(
        "--unit_generation_beam_size",
        type=int,
        help=(
            "Beam size for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=5,
    )
    parser.add_argument(
        "--unit_generation_max_len_a",
        type=int,
        help=(
            "`a` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=25,
    )
    parser.add_argument(
        "--unit_generation_max_len_b",
        type=int,
        help=(
            "`b` in `ax + b` for incremental unit decoding"
            "not applicable for the NAR T2U decoder."
        ),
        default=50,
    )
    parser.add_argument(
        "--unit_generation_ngram_blocking",
        type=bool,
        help=(
            "Enable ngram_repeat_block for incremental unit decoding."
            "This blocks hypotheses with repeating ngram tokens."
        ),
        default=False,
    )
    parser.add_argument(
        "--unit_generation_ngram_filtering",
        type=bool,
        help=(
            "If True, removes consecutive repeated ngrams"
            "from the decoded unit output."
        ),
        default=False,
    )
    parser.add_argument(
        "--text_unk_blocking",
        type=bool,
        help=(
            "If True, set penalty of UNK to inf in text generator "
            "to block unk output."
        ),
        default=False,
    )
    return parser


def set_generation_opts(
    args: Namespace,
) -> Tuple[SequenceGeneratorOptions, SequenceGeneratorOptions]:
    # Set text, unit generation opts.
    text_generation_opts = SequenceGeneratorOptions(
        beam_size=args.text_generation_beam_size,
        soft_max_seq_len=(
            args.text_generation_max_len_a,
            args.text_generation_max_len_b,
        ),
    )
    if args.text_unk_blocking:
        text_generation_opts.unk_penalty = torch.inf
    if args.text_generation_ngram_blocking:
        text_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )

    unit_generation_opts = SequenceGeneratorOptions(
        beam_size=args.unit_generation_beam_size,
        soft_max_seq_len=(
            args.unit_generation_max_len_a,
            args.unit_generation_max_len_b,
        ),
    )
    if args.unit_generation_ngram_blocking:
        unit_generation_opts.step_processor = NGramRepeatBlockProcessor(
            ngram_size=args.no_repeat_ngram_size
        )
    return text_generation_opts, unit_generation_opts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M4T inference on supported tasks using Translator."
    )
    parser.add_argument("input", type=str, help="Audio WAV file path or text input.")

    parser = add_inference_arguments(parser)
    args = parser.parse_args()
    if not args.task or not args.tgt_lang:
        raise Exception(
            "Please provide required arguments for evaluation -  task, tgt_lang"
        )

    if args.task.upper() in {"S2ST", "T2ST"} and args.output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    logger.info(f"Running inference on {device=} with {dtype=}.")

    translator = Translator(args.model_name, args.vocoder_name, device, dtype=dtype)

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    text_output, speech_output = translator.predict(
        args.input,
        args.task,
        args.tgt_lang,
        src_lang=args.src_lang,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=unit_generation_opts,
        unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
    )

    if speech_output is not None:
        logger.info(f"Saving translated audio in {args.tgt_lang}")
        torchaudio.save(
            args.output_path,
            speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
            sample_rate=speech_output.sample_rate,
        )
    logger.info(f"Translated text in {args.tgt_lang}: {text_output[0]}")


if __name__ == "__main__":
    main()
