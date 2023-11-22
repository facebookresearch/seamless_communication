# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Dict

from seamless_communication.models.vocoder.vocoder import Vocoder
from simuleval.agents import AgentStates, TextToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment


class VocoderAgent(TextToSpeechAgent):
    def __init__(self, vocoder: Vocoder, args: Namespace) -> None:
        super().__init__(args)
        self.sample_rate = args.sample_rate
        self.vocoder = vocoder
        self.tgt_lang = args.tgt_lang
        self.speaker_id = args.vocoder_speaker_id

    def policy(self, states: AgentStates) -> WriteAction:
        """
        The policy is always write if there are units
        """
        units = states.source

        if len(units) == 0 or len(units[0]) == 0:
            if states.source_finished:
                return WriteAction([], finished=True)
            else:
                return ReadAction()

        tgt_lang = states.tgt_lang if states.tgt_lang else self.tgt_lang
        u = units[0][0].tolist()
        wav_samples = self.vocoder(u, tgt_lang, self.speaker_id, dur_prediction=False)[
            0
        ][0].tolist()
        states.source = []

        return WriteAction(
            SpeechSegment(
                content=wav_samples,
                finished=states.source_finished,
                sample_rate=self.sample_rate,
                tgt_lang=tgt_lang,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--vocoder-speaker-id",
            type=int,
            required=False,
            default=-1,
            help="Vocoder speaker id",
        )

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: Dict[str, Any]) -> VocoderAgent:
        vocoder = kwargs.get("vocoder", None)
        assert isinstance(vocoder, Vocoder)
        return cls(vocoder, args)
