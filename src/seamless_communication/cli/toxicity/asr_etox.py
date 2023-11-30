# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import tempfile
import typing as tp
import torchaudio
from tqdm import tqdm
from seamless_communication.cli.eval_utils.compute_metrics import init_whisper_model
from seamless_communication.cli.eval_utils.lang_mapping import LANG3_LANG2
from seamless_communication.inference.translator import Modality
import torch

from pathlib import Path
from seamless_communication.inference import Translator
from fairseq2.data import Collater, DataPipeline, FileMapper
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.typing import DataType, Device

from seamless_communication.toxicity import load_etox_bad_word_checker

from whisper.model import Whisper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASR ETOX will compute the toxicity level of speech inputs."
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
        help="Language, language of the speech to transcribe",
        required=True,
    )
    parser.add_argument(
        "--audio_root_dir",
        type=str,
        help="Root directory for the audio filenames in the data file.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        help="Name of the column where the audiofile is listed in the input tsv.",
        default="audio",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help=(
            "Base model name (`seamlessM4T_medium`, "
            "`seamlessM4T_large`, `seamlessM4T_v2_large`), "
            " or whisper model, e.g. 'whisper_large'"
        ),
        default="seamlessM4T_v2_large",
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
    args, _unknown = parser.parse_known_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    whisper_model = None
    translator = None
    is_whisper = False

    if args.model_name.startswith("whisper_"):
        logger.info("loading whisper model.")
        _, model_name = args.model_name.split("_", maxsplit=1)
        whisper_model = init_whisper_model(device, model_name)
        is_whisper = True
    else:
        logger.info(f"loading {args.model_name} model.")
        translator = Translator(
            args.model_name,
            None,
            device,
            text_tokenizer=None,
            dtype=dtype,
            input_modality=Modality.SPEECH,
            output_modality=Modality.TEXT,
            apply_mintox=False,
        )

    logger.info("loading etox.")
    bad_word_checker = load_etox_bad_word_checker("mintox")

    pipeline = build_data_pipeline(
        data_file=args.data_file,
        audio_root_dir=args.audio_root_dir,
        batch_size=args.batch_size,
        is_whisper=is_whisper,
        device=device,
        dtype=dtype,
        n_parallel=args.n_parallel,
        audio_column=args.audio_column,
    )

    logger.info("running ASR-ETOX.")
    with open(args.output_file, "w", encoding="utf-8") as outf:
        print("text", "toxicity", "bad_words", file=outf, sep="\t")
        for example in tqdm(pipeline, unit="line"):
            texts = get_text(
                lang=args.lang,
                example=example,
                whisper_model=whisper_model,
                translator=translator,
                audio_column=args.audio_column,
            )
            for t in texts:
                bad_words = bad_word_checker.get_bad_words(
                    text=str(t),
                    lang=args.lang,
                )
                print(
                    t,
                    len(bad_words),
                    ",".join(bad_words),
                    file=outf,
                    sep="\t",
                )


def get_text(
    lang: str,
    example: tp.Dict[str, tp.Any],
    whisper_model: Whisper,
    translator: Translator,
    audio_column: str,
):
    if whisper_model:
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp:
            torchaudio.save(
                temp.name,
                example[audio_column]["data"]["waveform"]["seqs"][0]
                .transpose(0, 1)
                .cpu(),
                int(example[audio_column]["data"]["sample_rate"][0]),
                format="wav",
            )
            results = whisper_model.transcribe(
                temp.name,
                language=LANG3_LANG2[lang],
            )
            return [results["text"]]
    else:
        (text_output, _speech_output) = translator.predict(
            example[audio_column]["data"]["fbank"],
            "ASR",
            lang,
            src_lang=lang,
        )
        return text_output


def build_data_pipeline(
    data_file: Path,
    audio_root_dir: str,
    batch_size: int,
    is_whisper: bool,
    device: Device,
    dtype: DataType,
    audio_column: str = "audio",
    n_parallel: int = 4,
) -> DataPipeline:
    with data_file.open("r", encoding="utf-8") as f:
        header = f.readline().strip("\n").split("\t")

    split_tsv = StrSplitter(names=header)

    pipeline_builder = read_text(data_file, rtrim=True).skip(1).map(split_tsv)

    map_file = FileMapper(root_dir=audio_root_dir, cached_fd_count=10)

    pipeline_builder.map(
        map_file,
        selector=audio_column,
        num_parallel_calls=n_parallel,
    )

    decode_audio = AudioDecoder(dtype=torch.float32, device=device)

    convert_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
        device=device,
        dtype=dtype,
    )

    # get tensor in waveform
    steps = [decode_audio]
    if not is_whisper:
        # also get the fbanks
        steps.append(convert_to_fbank)

    pipeline_builder.map(
        steps,
        selector=f"{audio_column}.data",
        num_parallel_calls=n_parallel,
    )

    if is_whisper:
        # no batching for whisper
        pipeline_builder.bucket(bucket_size=batch_size)

    collate = Collater(pad_value=0, pad_to_multiple=1)

    pipeline_builder.map(collate, num_parallel_calls=n_parallel)

    pipeline_builder.prefetch(4)

    return pipeline_builder.and_return()


if __name__ == "__main__":
    main()
