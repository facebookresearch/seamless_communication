# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import sys

from seamless_communication.toxicity import load_etox_bad_word_checker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETOX will compute the toxicity level of text inputs (STDIN > STDOUT)."
    )
    parser.add_argument(
        "lang",
        type=str,
        help="Language, language of the speech to transcribe",
    )
    parser.add_argument(
        "input", nargs="?", type=argparse.FileType("r"), default=sys.stdin
    )
    parser.add_argument(
        "output", nargs="?", type=argparse.FileType("w"), default=sys.stdout
    )
    args, _unknown = parser.parse_known_args()

    bad_word_checker = load_etox_bad_word_checker("mintox")

    print("text", "toxicity", "bad_words", sep="\t", file=args.output)
    for line in args.input:
        l = line.rstrip()
        bad_words = bad_word_checker.get_bad_words(
            text=l,
            lang=args.lang,
        )
        print(l, len(bad_words), ",".join(bad_words), sep="\t", file=args.output)


if __name__ == "__main__":
    main()
