# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

"""
Script to create mExpresso Eng-XXX S2T dataset.
"""

import argparse
import logging
import multiprocessing as mp
import os
import pandas as pd
import pathlib
import re
import seamless_communication  # need this to load dataset cards
import torchaudio

from pathlib import Path
from tqdm import tqdm
from typing import List, Optional, Tuple

from fairseq2.assets import asset_store, download_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def multiprocess_map(
    a_list: list,
    func: callable,
    n_workers: Optional[int] = None,
    chunksize: int = 1,
    desc=None,
):
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, mp.cpu_count())
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(func, a_list, chunksize=chunksize),
                total=len(a_list),
                desc=desc,
            )
        )
    return results


def convert_to_16khz_wav(config: Tuple[str, str]) -> str:
    input_audio, output_audio = config
    input_wav, input_sr = torchaudio.load(input_audio)
    effects = [
        ["rate", "16000"],
        ["channels", "1"],
    ]
    wav, _ = torchaudio.sox_effects.apply_effects_tensor(
        input_wav, input_sr, effects=effects
    )
    os.makedirs(Path(output_audio).parent, exist_ok=True)
    torchaudio.save(
        output_audio, wav, sample_rate=16000, encoding="PCM_S", bits_per_sample=16
    )
    return output_audio


def build_en_manifest_from_oss(oss_root: Path, output_folder: Path) -> pd.DataFrame:
    # We only open source the following styles
    WHITELIST_STYLE = [
        "default",
        "default_emphasis",
        "default_essentials",
        "confused",
        "happy",
        "sad",
        "enunciated",
        "whisper",
        "laughing",
    ]

    results = []
    with open(oss_root / "read_transcriptions.txt") as fin:
        for line in fin:
            uid, text = line.strip().split("\t")
            sps = uid.split("_")
            oss_speaker = sps[0]
            style = "_".join(sps[1:-1])
            base_style = style.split("_")[0]
            if style not in WHITELIST_STYLE:
                continue
            # Normalize the text to remove <laugh> and <breath> etc
            text = re.sub(r" <.*?>", "", text)
            text = re.sub(r"<.*?> ", "", text)
            results.append(
                {
                    "id": uid,
                    "speaker": oss_speaker,
                    "text": text,
                    "orig_audio": (
                        oss_root
                        / "audio_48khz"
                        / "read"
                        / oss_speaker
                        / base_style
                        / "base"
                        / f"{uid}.wav"
                    ).as_posix(),
                    "label": style,
                }
            )

    df = pd.DataFrame(results)

    # Sanity checks
    # Check 1: audio files exists
    orig_audio_exists = df["orig_audio"].apply(lambda x: os.path.isfile(x))
    assert all(orig_audio_exists), df[~orig_audio_exists].iloc[0]["orig_audio"]

    # Convert 48kHz -> 16kHz
    target_audio_root = output_folder / "audio_16khz_wav"
    os.makedirs(target_audio_root, exist_ok=True)
    input_output_audios = [
        (
            row["orig_audio"],
            (target_audio_root / row["speaker"] / (row["id"] + ".wav")).as_posix(),
        )
        for i, row in df.iterrows()
    ]
    logger.info("converting from 48khz to mono 16khz")
    multiprocess_map(input_output_audios, convert_to_16khz_wav, chunksize=50)
    df.loc[:, "audio"] = [output_audio for _, output_audio in input_output_audios]
    audio_exists = df["audio"].apply(lambda x: os.path.isfile(x))
    assert all(audio_exists), df[~audio_exists].iloc[0]["audio"]
    output_manifest = f"{output_folder}/en_manifest.tsv"
    df.to_csv(output_manifest, sep="\t", quoting=3, index=None)
    logger.info(f"Output {len(df)} rows to {output_manifest}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare mExpresso Eng-XXX S2T manifest"
    )
    parser.add_argument(
        "output_folder",
        type=lambda p: pathlib.Path(p).resolve(),  # always convert to absolute path
        help="Output folder for the downsampled Expresso En audios and combined manifest. "
        "The output folder path will be expanded to absolute path.",
    )
    parser.add_argument(
        "--existing-expresso-root",
        type=str,
        help="Existing root folder if you have downloaded Expresso dataset. "
        "The folder path should include 'read_transcriptions.txt' and 'audio_48khz'",
    )
    args = parser.parse_args()

    mexpresso_card = asset_store.retrieve_card("mexpresso_text")
    mexpresso_root_path = download_manager.download_dataset(
        mexpresso_card.field("uri").as_uri(),
        "mExpresso_text",
    )
    logger.info(f"The mExpresso dataset is downloaded to {mexpresso_root_path}")
    mexpresso_path = mexpresso_root_path / "mexpresso_text"

    # downsample all English speech
    if args.existing_expresso_root is not None:
        logger.info(
            f"Re-use user manually downloaded Expresso from {args.existing_expresso_root}"
        )
        en_expresso_path = Path(args.existing_expresso_root)
    else:
        en_expresso_card = asset_store.retrieve_card("expresso")
        en_expresso_root_path = download_manager.download_dataset(
            en_expresso_card.field("uri").as_uri(),
            "Expresso",
        )
        logger.info(
            f"The English Expresso dataset is downloaded to {en_expresso_root_path}"
        )
        en_expresso_path = en_expresso_root_path / "expresso"
    en_expresso_folder = args.output_folder / "En_Expresso"
    en_expresso_df = build_en_manifest_from_oss(
        Path(en_expresso_path), en_expresso_folder
    )

    for subset in ["dev", "test"]:
        for lang in ["spa", "fra", "ita", "cmn", "deu"]:
            df = pd.read_csv(
                f"{mexpresso_path}/{subset}_mexpresso_{lang}.tsv", sep="\t", quoting=3
            ).rename(columns={"text": "tgt_text"})
            num_released_items = len(df)
            df = df.merge(
                en_expresso_df.rename(
                    columns={
                        "text": "src_text",
                        "audio": "src_audio",
                        "speaker": "src_speaker",
                    }
                ),
                on="id",
                how="inner",
            )
            assert (
                len(df) == num_released_items
            ), f"Missing items from downloaded En Expresso"
            df["src_lang"] = "eng"
            df["tgt_lang"] = lang
            # Check all the audio files exist
            assert all(os.path.isfile(audio) for audio in df["src_audio"].tolist())
            output_manifest_path = args.output_folder / f"{subset}_mexpresso_eng_{lang}.tsv"
            df[
                [
                    "id",
                    "src_audio",  # converted 16kHz audio path
                    "src_speaker",  # source speaker
                    "src_text",  # source text
                    "src_lang",  # source language id
                    "tgt_text",  # target text
                    "tgt_lang",  # target language id
                    "label",  # style of utterance
                ]
            ].to_csv(output_manifest_path, sep="\t", quoting=3, index=None)
            logger.info(f"Output {len(df)} rows to {output_manifest_path}")


if __name__ == "__main__":
    main()
