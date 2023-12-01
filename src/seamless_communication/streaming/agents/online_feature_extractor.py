# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
import torch

from argparse import ArgumentParser, Namespace
from typing import Any, List

from fairseq2.data.audio import WaveformToFbankConverter, WaveformToFbankInput

from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import Segment, SpeechSegment
from seamless_communication.streaming.agents.common import AgentStates


SHIFT_SIZE = 10
WINDOW_SIZE = 25
SAMPLE_RATE = 16000
FEATURE_DIM = 80


class FeatureStates(AgentStates):  # type: ignore
    def reset(self) -> None:
        super().reset()
        self.previous_residual_samples: List[float] = []
        self.tgt_lang = None

    def update_source(self, segment: Segment) -> None:
        """
        Update states from input segment
        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if self.tgt_lang is None and segment.tgt_lang is not None:
            self.tgt_lang = segment.tgt_lang
        if not segment.is_empty:
            self.source.append(segment.content)


class OnlineFeatureExtractorAgent(SpeechToSpeechAgent):  # type: ignore
    """
    Extract speech features on the fly.
    """

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.shift_size = args.shift_size
        self.window_size = args.window_size
        assert self.window_size >= self.shift_size

        self.sample_rate = args.sample_rate
        self.feature_dim = args.feature_dim
        self.num_samples_per_shift = int(self.shift_size * self.sample_rate / 1000)
        self.num_samples_per_window = int(self.window_size * self.sample_rate / 1000)
        self.len_ms_to_samples = lambda x: x * self.sample_rate / 1000

        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15 if args.denormalize else 1.0,
            standardize=False,
            device=args.device,
            dtype=args.dtype,
        )

    def build_states(self) -> FeatureStates:
        return FeatureStates()

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--shift-size",
            type=int,
            default=SHIFT_SIZE,
            help="Shift size of feature extraction window.",
        )
        parser.add_argument(
            "--window-size",
            type=int,
            default=WINDOW_SIZE,
            help="Window size of feature extraction window.",
        )
        parser.add_argument(
            "--feature-dim",
            type=int,
            default=FEATURE_DIM,
            help="Acoustic feature dimension.",
        )
        parser.add_argument(
            "--denormalize",
            action="store_true",
            help="denormalized to 16-bit signed integers",
        )

    def policy(self, states: FeatureStates) -> Action:
        if len(states.source) == 0:
            if states.source_finished:
                return WriteAction({}, finished=states.source_finished)
            else:
                return ReadAction()

        samples = states.source[-1]

        samples = states.previous_residual_samples + samples
        if len(samples) < self.num_samples_per_window:
            states.previous_residual_samples = samples
            return ReadAction()

        # num_frames is the number of frames from the new segment
        num_frames = math.floor(
            (len(samples) - self.len_ms_to_samples(self.window_size - self.shift_size))
            / self.num_samples_per_shift
        )

        # the number of frames used for feature extraction
        # including some part of the previous segment
        effective_num_samples = int(
            num_frames * self.len_ms_to_samples(self.shift_size)
            + self.len_ms_to_samples(self.window_size - self.shift_size)
        )

        input_samples = samples[:effective_num_samples]
        states.previous_residual_samples = samples[
            num_frames * self.num_samples_per_shift :
        ]

        data: WaveformToFbankInput = {
            "waveform": torch.tensor(input_samples).unsqueeze(0),
            "sample_rate": self.sample_rate,
        }

        output = self.convert_to_fbank(data)["fbank"]

        return WriteAction(
            SpeechSegment(
                content=output,
                tgt_lang=states.tgt_lang,
                finished=states.source_finished,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def from_args(cls, args: Any, **kwargs: Any) -> OnlineFeatureExtractorAgent:
        return cls(args)
