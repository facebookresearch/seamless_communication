# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any, Dict

import torch
from seamless_communication.models.vocoder.loader import load_vocoder_model
from seamless_communication.streaming.agents.common import AgentStates
from simuleval.agents import TextToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class VocoderAgent(TextToSpeechAgent):  # type: ignore
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        logger.info(
            f"Loading the Vocoder model: {args.vocoder_name} on device={args.device}, dtype={args.dtype}"
        )
        self.vocoder = load_vocoder_model(
            args.vocoder_name, device=args.device, dtype=args.dtype
        )
        self.vocoder.eval()

        self.sample_rate = args.sample_rate
        self.tgt_lang = args.tgt_lang
        self.speaker_id = args.vocoder_speaker_id

    @torch.inference_mode()
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
        u = units[0][0]

        wav = self.vocoder(u, tgt_lang, self.speaker_id, dur_prediction=False)
        states.source = []

        return WriteAction(
            SpeechSegment(
                content=wav[0][0].tolist(),
                finished=states.source_finished,
                sample_rate=self.sample_rate,
                tgt_lang=tgt_lang,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--vocoder-name",
            type=str,
            help="Vocoder name.",
            default="vocoder_v2",
        )
        parser.add_argument(
            "--vocoder-speaker-id",
            type=int,
            required=False,
            default=-1,
            help="Vocoder speaker id",
        )

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: Dict[str, Any]) -> VocoderAgent:
        return cls(args)
