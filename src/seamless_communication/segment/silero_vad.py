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
        #self.model.to("cuda")


    def segment_long_input(self, audio) -> None:
        """
        Split long input into chunks
        """
        max_segment_length_samples = self.chunk_size_sec * self.sample_rate
        pause_length_samples = self.pause_length * self.sample_rate

        speech_timestamps = self.get_speech_timestamps(audio, self.model, sampling_rate=self.sample_rate)

        segments = []
        current_segment = []

        for segment in speech_timestamps:
            start_samples = segment['start']
            end_samples = segment['end']

            if (current_segment and
                (end_samples - current_segment[0] > max_segment_length_samples or
                start_samples - current_segment[1] > pause_length_samples)):
                segments.append(current_segment)
                current_segment = []

            if not current_segment:
                current_segment = [start_samples, end_samples]
            else:
                current_segment[1] = end_samples
        if current_segment:
            segments.append(current_segment)
     
        return segments
      