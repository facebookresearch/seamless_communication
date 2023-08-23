# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import torch
import torchaudio
from seamless_communication.models.inference import Translator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="M4T inference on supported tasks using Translator."
    )
    parser.add_argument("input", type=str, help="Audio WAV file path or text input.")
    parser.add_argument("task", type=str, help="Task type")
    parser.add_argument(
        "tgt_lang", type=str, help="Target language to translate/transcribe into."
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        help="Source language, only required if input is text.",
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
    parser.add_argument(
        "--vocoder_name", type=str, help="Vocoder name", default="vocoder_36langs"
    )
    parser.add_argument(
        "--ngram-filtering",
        type=bool,
        help="Enable ngram_repeat_block (currently hardcoded to 4, during decoding) and ngram filtering over units (postprocessing)",
        default=False,
    )

    args = parser.parse_args()

    if args.task.upper() in {"S2ST", "T2ST"} and args.output_path is None:
        raise ValueError("output_path must be provided to save the generated audio")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
        logger.info(f"Running inference on the GPU in {dtype}.")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.info(f"Running inference on the CPU in {dtype}.")

    translator = Translator(args.model_name, args.vocoder_name, device, dtype)
    translated_text, wav, sr = translator.predict(
        args.input,
        args.task,
        args.tgt_lang,
        src_lang=args.src_lang,
        ngram_filtering=args.ngram_filtering,
    )

    if wav is not None and sr is not None:
        logger.info(f"Saving translated audio in {args.tgt_lang}")
        torchaudio.save(
            args.output_path,
            wav[0].cpu(),
            sample_rate=sr,
        )
    logger.info(f"Translated text in {args.tgt_lang}: {translated_text}")


if __name__ == "__main__":
    main()
