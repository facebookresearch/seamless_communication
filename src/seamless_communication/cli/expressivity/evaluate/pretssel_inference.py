# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torchaudio
from fairseq2.data import Collater, DataPipeline, FileMapper
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.typing import DataType, Device
from sacrebleu.metrics import BLEU  # type: ignore[attr-defined]
from torch import Tensor
from tqdm import tqdm

from seamless_communication.cli.expressivity.evaluate.pretssel_inference_helper import (
    PretsselGenerator,
)
from seamless_communication.cli.m4t.evaluate.evaluate import (
    adjust_output_for_corrupted_inputs,
    count_lines,
)
from seamless_communication.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)
from seamless_communication.inference import BatchedSpeechOutput, Translator
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.store import add_gated_assets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def build_data_pipeline(
    args: Namespace,
    device: Device,
    dtype: DataType,
    gcmvn_mean: Tensor,
    gcmvn_std: Tensor,
) -> DataPipeline:
    with open(args.data_file, "r") as f:
        header = f.readline().strip("\n").split("\t")
        assert (
            args.audio_field in header
        ), f"Input file does not contain {args.audio_field} field"

    n_parallel = 4

    split_tsv = StrSplitter(names=header)

    pipeline_builder = read_text(args.data_file, rtrim=True).skip(1).map(split_tsv)

    assert args.audio_root_dir is not None

    map_file = FileMapper(root_dir=args.audio_root_dir, cached_fd_count=10)

    pipeline_builder.map(
        map_file, selector=args.audio_field, num_parallel_calls=n_parallel
    )

    decode_audio = AudioDecoder(dtype=torch.float32, device=device)

    convert_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=device,
        dtype=dtype,
    )

    def normalize_fbank(data: WaveformToFbankOutput) -> WaveformToFbankOutput:
        fbank = data["fbank"]
        std, mean = torch.std_mean(fbank, dim=0)
        data["fbank"] = fbank.subtract(mean).divide(std)
        data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
        return data

    pipeline_builder.map(
        [decode_audio, convert_to_fbank, normalize_fbank],
        selector=f"{args.audio_field}.data",
        num_parallel_calls=n_parallel,
    )

    pipeline_builder.bucket(bucket_size=args.batch_size)

    collate = Collater(pad_value=0, pad_to_multiple=1)

    pipeline_builder.map(collate, num_parallel_calls=n_parallel)

    pipeline_builder.prefetch(4)

    return pipeline_builder.and_return()


def main() -> None:
    parser = argparse.ArgumentParser(description="Running SeamlessExpressive inference")
    parser.add_argument(
        "data_file", type=Path, help="Data file (.tsv) to be evaluated."
    )

    parser = add_inference_arguments(parser)
    param = parser.add_argument(
        "--gated-model-dir",
        type=Path,
        required=False,
        help="SeamlessExpressive model directory.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Inference batch size.",
        default=4,
    )
    parser.add_argument(
        "--audio_root_dir",
        type=Path,
        help="Root directory for the audio filenames in the data file.",
        default="",
    )
    parser.add_argument(
        "--audio_field",
        type=str,
        help="Field that includes the input audio file paths.",
        default="src_audio",
    )
    parser.add_argument(
        "--ref_field",
        type=str,
        help="Reference target text field to compute the BLEU score against.",
        default=None,
    )
    parser.add_argument(
        "--duration_factor",
        type=float,
        help="The duration factor for NAR T2U model.",
        default=1.0,
    )
    parser.add_argument(
        "--output_result_tsv",
        type=bool,
        help="Whether to output results in tsv format (for full-blown evaluation)",
        default=True,
    )
    args = parser.parse_args()

    if args.gated_model_dir:
        add_gated_assets(args.gated_model_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    unit_tokenizer = load_unity_unit_tokenizer(args.model_name)

    _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(args.vocoder_name)
    gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
    gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    pipeline = build_data_pipeline(args, device, dtype, gcmvn_mean, gcmvn_std)

    translator = Translator(
        args.model_name,
        vocoder_name_or_card=None,
        device=device,
        dtype=dtype,
    )

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    pretssel_generator = PretsselGenerator(
        args.vocoder_name,
        vocab_info=unit_tokenizer.vocab_info,
        device=device,
        dtype=dtype,
    )

    total_steps = count_lines(args.data_file) - 1
    progress_bar = tqdm(total=total_steps)

    output_path = args.output_path / args.data_file.stem
    output_path.mkdir(parents=True, exist_ok=True)

    waveforms_dir = output_path / "waveform"
    waveforms_dir.mkdir(parents=True, exist_ok=True)

    hyps = []
    refs = []
    audio_hyps = []

    with contextlib.ExitStack() as stack:
        hyp_file = stack.enter_context(
            open(output_path / f"text_output-{args.data_file.stem}.txt", "w")
        )
        unit_file = stack.enter_context(
            open(output_path / f"unit_output-{args.data_file.stem}.txt", "w")
        )

        sample_id = 0
        for example in pipeline:
            valid_sequences: Optional[Tensor] = None
            src = example[args.audio_field]["data"]["fbank"]
            # Skip corrupted audio tensors.
            valid_sequences = ~torch.any(
                torch.any(torch.isnan(src["seqs"]), dim=1), dim=1
            )
            if not valid_sequences.all():
                logger.warning(
                    f"Sample IDs {sample_id} to {sample_id + args.batch_size} has some corrupted input."
                )
                src["seqs"] = src["seqs"][valid_sequences]
                src["seq_lens"] = src["seq_lens"][valid_sequences]

            # Skip performing inference when the input is entirely corrupted.
            if src["seqs"].numel() > 0:
                prosody_encoder_input = example[args.audio_field]["data"]["gcmvn_fbank"]
                text_output, unit_output = translator.predict(
                    src,
                    args.task,
                    args.tgt_lang,
                    src_lang=args.src_lang,
                    text_generation_opts=text_generation_opts,
                    unit_generation_opts=unit_generation_opts,
                    unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
                    duration_factor=args.duration_factor,
                    prosody_encoder_input=prosody_encoder_input,
                )

                assert unit_output is not None
                speech_output = pretssel_generator.predict(
                    unit_output.units,
                    tgt_lang=args.tgt_lang,
                    prosody_encoder_input=prosody_encoder_input,
                )

            else:
                text_output = []
                speech_output = BatchedSpeechOutput(units=[], audio_wavs=[])

            if valid_sequences is not None and not valid_sequences.all():
                text_output, speech_output = adjust_output_for_corrupted_inputs(  # type: ignore[assignment]
                    valid_sequences,
                    text_output,
                    speech_output,
                )

            hyps += [str(s) for s in text_output]
            if args.ref_field is not None and args.ref_field in example:
                refs += [str(s) for s in example[args.ref_field]]

            for i in range(len(text_output)):
                t = text_output[i]
                idx = str(example["id"][i])
                hyp_file.write(f"{t}\n")

                u = speech_output.units[i]
                str_units = [str(i) for i in u]
                unit_file.write(" ".join(str_units) + "\n")
                torchaudio.save(
                    waveforms_dir / f"{idx}_pred.wav",
                    speech_output.audio_wavs[i][0].to(torch.float32).cpu(),
                    sample_rate=speech_output.sample_rate,
                )
                audio_hyps.append((waveforms_dir / f"{idx}_pred.wav").as_posix())

                sample_id += 1
                progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Processed {len(hyps)} hyps, {len(refs)} refs")

    if args.output_result_tsv:
        output_tsv_file = output_path / f"generate-{args.data_file.stem}.tsv"
        output_tsv = pd.read_csv(args.data_file, quoting=3, sep="\t")
        text_out = []
        with open(hyp_file.name) as file:
            for line in file:
                text_out.append(line.strip())

        unit_out = []
        with open(unit_file.name) as file:
            for line in file:
                unit_out.append(line.strip())

        output_tsv["hypo_audio"] = audio_hyps
        output_tsv["s2t_out"] = text_out
        output_tsv["orig_unit"] = unit_out
        output_tsv.to_csv(output_tsv_file, quoting=3, sep="\t", index=False)
        logger.info(f"Output results in {output_tsv_file}")


if __name__ == "__main__":
    main()
