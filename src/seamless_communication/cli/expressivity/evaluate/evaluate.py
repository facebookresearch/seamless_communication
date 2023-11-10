# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from fairseq2.data import Collater, CString, DataPipeline, FileMapper
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)
from fairseq2.data.text import StrSplitter, TextTokenizer, read_text
from fairseq2.data.typing import PathLike, StringLike
from fairseq2.generation import SequenceGeneratorOptions
from fairseq2.typing import DataType, Device
from sacrebleu.metrics import BLEU  # type: ignore[attr-defined]
from torch import Tensor
from tqdm import tqdm

from seamless_communication.cli.m4t.predict import (
    add_inference_arguments,
    set_generation_opts,
)
from seamless_communication.inference import BatchedSpeechOutput, Modality, Translator
from seamless_communication.models.unity import load_unity_text_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    task: str
    """String representing the task. Valid choices are
    "S2ST", "S2TT", "T2ST", "T2TT", "ASR"."""

    output_modality: Modality
    """The output modality of the task."""

    model_name: str
    """The name of the S2T UnitY model."""

    data_file: Path
    """The pathname of the test TSV data file."""

    audio_root_dir: Optional[Path]
    """The pathname of the directory under which
    audio files are stored."""

    target_lang: str
    """The target translation language."""

    source_lang: Optional[str]
    """The source language."""

    batch_size: int
    """The batch size for model input."""

    device: Device
    """The device on which to run inference."""

    dtype: DataType
    """The data type with which to run inference."""

    output_path: Path
    """The pathname of the output directory to save
    the evaluation results."""

    ref_field: str
    """The reference target text field to compute
    the BLEU score against."""

    text_generation_opts: SequenceGeneratorOptions
    """Text generation hyperparameters."""

    unit_generation_opts: Optional[SequenceGeneratorOptions]
    """Unit generation hyperparameters, not applicable
    for the NAR T2U decoder."""

    unit_generation_ngram_filtering: bool
    """If True, removes consecutive repeating ngrams
    from the decoded unit output."""

    gcmvn_stats: Optional[PathLike] = None
    """the stats for gcmvn, used by Prosody Encoder"""


def count_lines(filename: Path) -> int:
    result = subprocess.run(["wc", "-l", filename], stdout=subprocess.PIPE)
    return int(result.stdout.decode().split()[0])


def build_data_pipeline(
    ctx: EvalContext,
    text_tokenizer: TextTokenizer,
) -> DataPipeline:
    with open(ctx.data_file, "r") as f:
        header = f.readline().strip("\n").split("\t")

    # TODO: This will be soon auto-tuned. Right now hand-tuned for devfair.
    n_parallel = 4

    split_tsv = StrSplitter(names=header)

    if ctx.gcmvn_stats is not None:
        if isinstance(ctx.gcmvn_stats, CString):
            ctx.gcmvn_stats = str(ctx.gcmvn_stats)
        gcmvn_stats: Dict[str, np.ndarray] = np.load(ctx.gcmvn_stats)  # type: ignore[type-arg]
        gcmvn_mean = torch.tensor(
            gcmvn_stats["mean"], device=ctx.device, dtype=ctx.dtype
        )
        gcmvn_std = torch.tensor(gcmvn_stats["std"], device=ctx.device, dtype=ctx.dtype)

    pipeline_builder = read_text(ctx.data_file, rtrim=True).skip(1).map(split_tsv)

    assert ctx.audio_root_dir is not None

    map_file = FileMapper(root_dir=ctx.audio_root_dir, cached_fd_count=10)

    pipeline_builder.map(map_file, selector="audio", num_parallel_calls=n_parallel)

    decode_audio = AudioDecoder(dtype=torch.float32, device=ctx.device)

    convert_to_fbank = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=False,
        device=ctx.device,
        dtype=ctx.dtype,
    )

    def normalize_fbank(data: WaveformToFbankOutput) -> WaveformToFbankOutput:
        fbank = data["fbank"]
        std, mean = torch.std_mean(fbank, dim=0)
        data["fbank"] = fbank.subtract(mean).divide(std)
        if ctx.gcmvn_stats is not None:
            data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
        return data

    pipeline_builder.map(
        [decode_audio, convert_to_fbank, normalize_fbank],
        selector="audio.data",
        num_parallel_calls=n_parallel,
    )

    pipeline_builder.bucket(bucket_size=ctx.batch_size)

    collate = Collater(pad_value=0, pad_to_multiple=1)

    pipeline_builder.map(collate, num_parallel_calls=n_parallel)

    pipeline_builder.prefetch(4)

    return pipeline_builder.and_return()


def adjust_output_for_corrupted_inputs(
    valid_sequences: Tensor,
    text_output: List[StringLike],
    speech_output: Optional[BatchedSpeechOutput],
) -> Tuple[List[StringLike], Optional[BatchedSpeechOutput]]:
    adjusted_text_output: List[StringLike] = []
    adjusted_speech_output: Optional[BatchedSpeechOutput] = None

    if speech_output is not None:
        assert (
            len(text_output)
            == len(speech_output.units)
            == len(speech_output.audio_wavs)
        )
        adjusted_speech_output = BatchedSpeechOutput(units=[], audio_wavs=[])

    batch_counter = 0
    for is_valid in valid_sequences:
        if is_valid:
            adjusted_text_output.append(text_output[batch_counter])
            if speech_output is not None:
                assert adjusted_speech_output is not None
                adjusted_speech_output.units.append(speech_output.units[batch_counter])
                adjusted_speech_output.audio_wavs.append(
                    speech_output.audio_wavs[batch_counter]
                )
            batch_counter += 1
        else:
            # For the corrupted inputs, we save the following dummy outputs:
            # empty string for text, empty list for units, 1 second of silence for audio.
            adjusted_text_output.append("")
            if adjusted_speech_output is not None:
                sample_rate = adjusted_speech_output.sample_rate
                adjusted_speech_output.units.append([])
                adjusted_speech_output.audio_wavs.append(
                    torch.zeros(sample_rate).unsqueeze(0).unsqueeze(0)
                )
    return (
        adjusted_text_output,
        adjusted_speech_output,
    )


def run_eval(
    translator: Translator, text_tokenizer: TextTokenizer, ctx: EvalContext
) -> None:
    pipeline = build_data_pipeline(ctx, text_tokenizer)

    total_steps = count_lines(ctx.data_file) - 1
    progress_bar = tqdm(total=total_steps)

    output_path = ctx.output_path / ctx.data_file.stem
    output_path.mkdir(parents=True, exist_ok=True)

    if ctx.output_modality == Modality.SPEECH:
        waveforms_dir = output_path / f"waveform_{ctx.data_file.stem}"
        waveforms_dir.mkdir(parents=True, exist_ok=True)

    hyps = []
    refs = []

    with contextlib.ExitStack() as stack:
        hyp_file = stack.enter_context(
            open(output_path / f"text_output-{ctx.data_file.stem}.txt", "w")
        )
        if ctx.output_modality == Modality.SPEECH:
            unit_file = stack.enter_context(
                open(output_path / f"unit_output-{ctx.data_file.stem}.txt", "w")
            )

        sample_id = 0
        for example in pipeline:
            valid_sequences: Optional[Tensor] = None
            src = example["audio"]["data"]["fbank"]
            # Skip corrupted audio tensors.
            valid_sequences = ~torch.any(
                torch.any(torch.isnan(src["seqs"]), dim=1), dim=1
            )
            if not valid_sequences.all():
                logger.warning(
                    f"Sample IDs {sample_id} to {sample_id + ctx.batch_size} has some corrupted input."
                )
                src["seqs"] = src["seqs"][valid_sequences]
                src["seq_lens"] = src["seq_lens"][valid_sequences]

            # Skip performing inference when the input is entirely corrupted.
            if src["seqs"].numel() > 0:
                (
                    text_output,
                    speech_output,
                ) = translator.predict(
                    src,
                    ctx.task,
                    ctx.target_lang,
                    src_lang=ctx.source_lang,
                    text_generation_opts=ctx.text_generation_opts,
                    unit_generation_opts=ctx.unit_generation_opts,
                    unit_generation_ngram_filtering=ctx.unit_generation_ngram_filtering,
                    gcmvn_fbank=example["audio"]["data"].get("gcmvn_fbank", None),
                )
            else:
                text_output = []
                if ctx.output_modality == Modality.SPEECH:
                    speech_output = BatchedSpeechOutput(units=[], audio_wavs=[])
                else:
                    speech_output = None

            if valid_sequences is not None and not valid_sequences.all():
                (
                    text_output,
                    speech_output,
                ) = adjust_output_for_corrupted_inputs(
                    valid_sequences,
                    text_output,
                    speech_output,
                )

            hyps += [str(s) for s in text_output]
            refs += [str(s) for s in example[ctx.ref_field]]

            for i in range(len(text_output)):
                t = text_output[i]
                hyp_file.write(f"{t}\n")

                if ctx.output_modality == Modality.SPEECH:
                    assert speech_output is not None
                    u = speech_output.units[i]
                    str_units = [str(i) for i in u]
                    unit_file.write(" ".join(str_units) + "\n")
                    torchaudio.save(
                        waveforms_dir / f"{sample_id}_pred.wav",
                        speech_output.audio_wavs[i][0].to(torch.float32).cpu(),
                        sample_rate=speech_output.sample_rate,
                    )

                sample_id += 1
                progress_bar.update(1)

    progress_bar.close()
    logger.info(f"Processed {len(hyps)} hyps, {len(refs)} refs")

    assert len(hyps) == len(refs)
    if len(hyps) > 0:
        if ctx.target_lang in ("cmn", "jpn", "lao", "mya", "tha"):
            tokenizer = "char"
        else:
            tokenizer = "13a"

        bleu = BLEU(tokenize=tokenizer)
        score = bleu.corpus_score(hyps, [refs])
        bleu_filename = output_path / f"{ctx.data_file.stem}_text_output_bleu.json"
        with open(bleu_filename, "w") as f:
            f.write(score.format(signature=str(bleu.get_signature()), is_json=True))
        logger.info(score.format(signature=bleu.get_signature()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expressivity evaluation for tasks supported by Translator."
    )
    parser.add_argument("data_file", type=str, help="Data file (.tsv) to be evaluated.")

    parser = add_inference_arguments(parser)
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Inference batch size.",
        default=4,
    )
    parser.add_argument(
        "--audio_root_dir",
        type=str,
        help="Root directory for the audio filenames in the data file.",
        default="",
    )
    parser.add_argument(
        "--ref_field",
        type=str,
        help="Reference target text field to compute the BLEU score against.",
        default="tgt_text",
    )
    parser.add_argument(
        "--gcmvn_stats",
        type=str,
        help="The path to gcmvn fbank stats, if provided, the DataPipeline'd have another copy of gcmvn fbank features (for P2V enc)",
        default=None,
    )
    args = parser.parse_args()

    input_modality, output_modality = Translator.get_modalities_from_task_str(args.task)

    if input_modality == Modality.SPEECH and not Path(args.audio_root_dir).exists():
        raise ValueError(
            f"Invalid audio_root_dir: {args.audio_root_dir} for speech input."
        )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    text_tokenizer = load_unity_text_tokenizer(args.model_name)

    # TODO: Avoid loading the T2U model, vocoder when the output
    # modality is text.
    translator = Translator(
        args.model_name,
        args.vocoder_name,
        device,
        text_tokenizer=text_tokenizer,
        dtype=dtype,
    )

    text_generation_opts, unit_generation_opts = set_generation_opts(args)

    logger.info(f"{text_generation_opts=}")
    logger.info(f"{unit_generation_opts=}")
    logger.info(
        f"unit_generation_ngram_filtering={args.unit_generation_ngram_filtering}"
    )

    # fmt: off
    ctx = EvalContext(
        task=args.task,
        output_modality=output_modality,
        model_name=args.model_name,
        data_file=Path(args.data_file),
        audio_root_dir=Path(args.audio_root_dir),
        target_lang=args.tgt_lang,
        source_lang=args.src_lang,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
        ref_field=args.ref_field,
        text_generation_opts=text_generation_opts,
        unit_generation_opts=unit_generation_opts,
        unit_generation_ngram_filtering=args.unit_generation_ngram_filtering,
        output_path=Path(args.output_path),
        gcmvn_stats=args.gcmvn_stats,
    )
    # fmt: on
    logger.info(f"Running inference on {device=} with {dtype=}, {ctx.batch_size=}.")

    run_eval(translator, text_tokenizer, ctx)


if __name__ == "__main__":
    main()
