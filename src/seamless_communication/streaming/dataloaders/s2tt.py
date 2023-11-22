# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import subprocess
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import Collater, DataPipeline, FileMapper
from fairseq2.data.text.converters import StrSplitter
from fairseq2.data.text.text_reader import read_text
from simuleval.data.dataloader import register_dataloader
from simuleval.data.dataloader.dataloader import IterableDataloader
from simuleval.data.dataloader.s2t_dataloader import SpeechToTextDataloader


@dataclass
class SoundFileInfo:
    samplerate: float
    path: str

    def __repr__(self) -> str:
        return "\n".join([f"samplerate: {str(self.samplerate)}", f"path: {self.path}"])


def count_lines(filename: Path) -> int:
    result = subprocess.run(["wc", "-l", filename], stdout=subprocess.PIPE)
    return int(result.stdout.decode().split()[0]) - 1


@register_dataloader("fairseq2_s2tt")
class SimulEvalSpeechToTextDataloader(SpeechToTextDataloader, IterableDataloader):  # type: ignore
    def __init__(self, data_pipeline: DataPipeline, args: Namespace) -> None:
        self.args = args
        self.data_file: Path = Path(getattr(self.args, "data_file", ""))
        if not self.data_file.exists():
            raise ValueError(f"data_file: {self.data_file} does not exist.")
        self.start_index: int = getattr(self.args, "start_index", 0)
        self.end_index: int = getattr(self.args, "end_index", -1)
        self.data_pipeline = data_pipeline
        self.data_itr = iter(self.data_pipeline)
        self.cur_index = self.start_index - 1

    def __iter__(self) -> SimulEvalSpeechToTextDataloader:
        return self

    def __next__(self) -> SimulEvalSpeechToTextDataloader:
        if self.cur_index >= self.end_index - 1:
            raise StopIteration
        self.item = next(self.data_itr)
        self.cur_index += 1
        return self

    def reset(self) -> None:
        self.cur_index = 0
        self.data_pipeline.reset()

    def __len__(self) -> int:
        if self.end_index > 0:
            return self.end_index - self.start_index
        self.end_index = count_lines(self.data_file)
        return self.end_index - self.start_index

    def get_source(self, index: Optional[int] = None) -> List[float]:
        source: List[float] = (
            self.item["audio"]["data"]["waveform"]["seqs"].squeeze().tolist()
        )
        return source

    def get_target(self, index: Optional[int] = None) -> str:
        return str(self.item[self.args.ref_field][0])

    def get_tgt_lang(self, index: Optional[int] = None) -> Optional[str]:
        if self.args.tgt_lang:
            tgt_lang: str = self.args.tgt_lang
            return tgt_lang

        tgt_lang = self.item.get("tgt_lang")
        return str(tgt_lang[0]) if tgt_lang else None

    def get_source_audio_info(self, index: Optional[int] = None) -> SoundFileInfo:
        samplerate = self.item["audio"]["data"]["sample_rate"][0]
        path = f'{self.args.audio_root_dir}/{str(self.item["audio"]["path"][0])}'
        return SoundFileInfo(samplerate, path)

    def get_source_audio_path(self, index: Optional[int] = None) -> str:
        return str(self.item["audio"]["path"][0])

    @classmethod
    def from_args(cls, args: Namespace) -> SimulEvalSpeechToTextDataloader:
        with open(args.data_file, "r") as f:
            header = f.readline().strip("\n").split("\t")

        split_tsv = StrSplitter(names=header)

        start_index: int = getattr(args, "start_index", 0)

        pipeline_builder = (
            read_text(args.data_file, rtrim=True).skip(1 + start_index).map(split_tsv)
        )

        map_file = FileMapper(root_dir=args.audio_root_dir, cached_fd_count=10)

        pipeline_builder.map(map_file, selector="audio")

        device = getattr(args, "device", None)
        assert device is not None

        decode_audio = AudioDecoder(dtype=torch.float32, device=torch.device(device))

        pipeline_builder.map(
            decode_audio,
            selector="audio.data",
        )

        pipeline_builder.map(
            lambda x: F.layer_norm(x, x.shape),
            selector="audio.data.waveform",
        )

        collate = Collater(pad_value=0, pad_to_multiple=1)

        pipeline_builder.map(collate)

        pipeline_builder.prefetch(1)

        data_pipeline = pipeline_builder.and_return()

        return cls(data_pipeline, args)

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--data-file",
            type=str,
            required=True,
            help="Data file (.tsv) to be evaluated.",
        )
        parser.add_argument(
            "--audio-root-dir",
            type=str,
            help="Root directory for the audio filenames in the data file.",
            default="",
        )
        parser.add_argument(
            "--ref-field",
            type=str,
            help="Reference target text field to compute the BLEU score against.",
            default="tgt_text",
        )
        parser.add_argument(
            "--source-segment-size",
            type=int,
            default=1,
            help="Source segment size, For text the unit is # token, for speech is ms",
        )
        parser.add_argument(
            "--tgt-lang", type=str, help="Target language to translate/transcribe into."
        )
