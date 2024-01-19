# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import sys

import torch
from seamless_communication.toxicity.mutox.loader import load_mutox_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

CPU_DEVICE = torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mutox Text will compute a toxicity score for each sentence it is passed."
    )

    parser.add_argument(
        "lang",
        type=str,
        help="Language of the input text, nllb format with script.",
    )
    parser.add_argument(
        "input", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument(
        "output", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Inference batch size.",
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
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    )

    classifier = load_mutox_model(
        "mutox",
        device=device,
        dtype=dtype,
    ).eval()

    def write_result(batch):
        emb = t2vec_model.predict(batch, source_lang=args.lang)
        scores = classifier(emb.half())
        for s, t in zip(scores, batch):
            print(t, s.item(), sep="\t", file=args.output)

    with torch.inference_mode():
        print("text", "score", sep="\t", file=args.output)
        batch = []
        for line in args.input:
            batch.append(line.rstrip())
            if len(batch) >= args.batch_size:
                write_result(batch)
                batch = []

        if len(batch):
            write_result(batch)


if __name__ == "__main__":
    main()
