# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from pathlib import Path
from argparse import ArgumentParser, Namespace
from os import SEEK_END
from typing import Any, List, Optional, Union
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

class SileroVADSegmenter:  # type: ignore
    def __init__(self, args: Namespace) -> None:
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )

        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        self.sample_rate = getattr(args, 'sample_rate', 16000)
        self.chunk_size_sec = getattr(args, 'chunk_size_sec', 10)
        self.pause_length = getattr(args, 'pause_length', 0.5)


    def segment_long_input(self, audio: str) -> None:
        """
        Split long input into chunks
        """
        max_segment_length_samples = self.chunk_size_sec * self.sample_rate # max segment length is 10 seconds
        pause_length_samples = self.pause_length * self.sample_rate

        if isinstance(audio, str):
            audio = self.read_audio(audio, sampling_rate=self.sample_rate)
        speech_timestamps = self.get_speech_timestamps(audio, self.model, self.sample_rate)

        segments = []
        current_segment = []

        # if adding the current speech segment would make the current segment too long,
        # or if the pause between the current speech segment and the next one is long enough,
        # add the current segment to the segments list
        for segment in speech_timestamps:
            start_samples = segment["start"]
            end_samples = segment["end"]

            if (
                current_segment
                and (end_samples - current_segment[0] > max_segment_length_samples 
                     or start_samples - current_segment[1] > pause_length_samples)
            ):
                segments.append(current_segment)
                current_segment = []

            if not current_segment:
                current_segment = [start_samples, end_samples]
            else:
                current_segment[1] = end_samples

        if current_segment:
            segments.append(current_segment)

        segmented_audios = []  

        for i, (start, end) in enumerate(segments):
            segment_timestamps = [{'start': start, 'end': end}]
            segmented_audio = self.collect_chunks(segment_timestamps, audio)
            segmented_audios.append(segmented_audio)

        return segmented_audios
      