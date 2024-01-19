# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse

import torch
from tqdm import tqdm
from pathlib import Path

from sonar.inference_pipelines.speech import (
    SpeechInferenceParams,
)
from seamless_communication.toxicity.mutox.speech_pipeline import (
    MutoxSpeechClassifierPipeline,
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mutox speech will compute a toxicity score for each speech segment it is provided."
    )
    parser.add_argument(
        "data_file",
        type=Path,
        help="Path to the input TSV manifest that list the audio files.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to a TSV file where to save the results.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        help="Language, language of the speech being passed as input, three letter code",
        required=True,
    )
    parser.add_argument(
        "--audio_root_dir",
        type=str,
        help="Root directory for the audio filenames in the data file.",
    )
    parser.add_argument(
        "--audio_path_index",
        type=int,
        help="Index of the column where the audiofile is listed in the input tsv.",
        default="audio",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Inference batch size.",
        default=4,
    )
    parser.add_argument(
        "--n_parallel",
        type=int,
        help="Number of data loading in parallel.",
        default=4,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="name of the device to use with torch.",
        required=False,
    )
    args, _unknown = parser.parse_known_args()

    if args.device is not None:
        device = torch.device(args.device)
        dtype = torch.float32
        if device.type == "cuda":
            dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
        logger.info("using cuda:0, %s", dtype)
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        logger.info("no gpu, using cpu")

    logger.info("loading models.")

    pipeline_builder = MutoxSpeechClassifierPipeline.load_model_from_name(
        mutox_classifier_name="mutox",
        encoder_name=f"sonar_speech_encoder_{args.lang}",
        device=device,
    )

    pipeline = pipeline_builder.build_pipeline(
        SpeechInferenceParams(
            data_file=args.data_file,
            audio_root_dir=args.audio_root_dir,
            audio_path_index=args.audio_path_index,
            target_lang=args.lang,
            batch_size=args.batch_size,
            pad_idx=0,
            device=device,
            fbank_dtype=torch.float32,
            n_parallel=args.n_parallel,
        )
    )

    logger.info("processing.")

    with open(args.output_file, "w", encoding="utf-8") as outf:
        print(
            "input_audio_path",
            "score",
            sep="\t",
            file=outf,
        )
        for example in tqdm(pipeline):
            ex = example["audio"]
            for idx, path in enumerate(ex["path"]):
                print(
                    str(path),
                    ex["data"][idx].item(),
                    sep="\t",
                    file=outf,
                )

    logger.info(f"Done, outputs are in {args.output_file}.")


if __name__ == "__main__":
    main()
