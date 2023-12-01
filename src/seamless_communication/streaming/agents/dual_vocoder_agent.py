# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations
import copy

import logging
from argparse import ArgumentParser, Namespace
from typing import Dict, Any

from simuleval.agents import TextToSpeechAgent
from seamless_communication.streaming.agents.common import AgentStates
from simuleval.data.segments import Segment
from simuleval.agents.actions import Action

from seamless_communication.streaming.agents.pretssel_vocoder import (
    PretsselVocoderAgent,
)
from seamless_communication.streaming.agents.online_vocoder import VocoderAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class DualVocoderStates(AgentStates):
    def __init__(
        self, vocoder_states: AgentStates, expr_vocoder_states: AgentStates
    ) -> None:
        self.vocoder_states = vocoder_states
        self.expr_vocoder_states = expr_vocoder_states
        self.config: Dict[str, Any] = {}

    @property
    def target_finished(self):  # type: ignore
        return (
            self.vocoder_states.target_finished
            or self.expr_vocoder_states.target_finished
        )

    def reset(self) -> None:
        self.vocoder_states.reset()
        self.expr_vocoder_states.reset()
        self.config = {}

    def update_source(self, segment: Segment) -> None:
        self.vocoder_states.update_config(segment.config)
        self.vocoder_states.update_source(segment)
        self.expr_vocoder_states.update_config(segment.config)
        self.expr_vocoder_states.update_source(segment)

    def update_target(self, segment: Segment) -> None:
        self.vocoder_states.update_target(segment)
        self.expr_vocoder_states.update_target(segment)


class DualVocoderAgent(TextToSpeechAgent):  # type: ignore
    def __init__(
        self,
        args: Namespace,
        vocoder: VocoderAgent,
        expr_vocoder: PretsselVocoderAgent,
    ) -> None:
        self.vocoder = vocoder
        self.expr_vocoder = expr_vocoder
        super().__init__(args)
        self.expressive = args.expressive

    def build_states(self) -> DualVocoderStates:
        return DualVocoderStates(
            self.vocoder.build_states(), self.expr_vocoder.build_states()
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        PretsselVocoderAgent.add_args(parser)
        VocoderAgent.add_args(parser)
        parser.add_argument(
            "--expr-vocoder-name",
            type=str,
            required=True,
            help="expressive vocoder name - vocoder_pretssel or vocoder_pretssel_16khz",
        )
        parser.add_argument(
            "--expressive",
            action="store_true",
            help="Whether to use expressive vocoder (overridable in segment.config)",
        )

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: Dict[str, Any]) -> DualVocoderAgent:
        vocoder = VocoderAgent.from_args(args)
        expr_args = copy.deepcopy(args)
        expr_args.vocoder_name = args.expr_vocoder_name
        expr_vocoder = PretsselVocoderAgent.from_args(expr_args)
        return cls(args, vocoder, expr_vocoder)

    def policy(self, states: AgentStates) -> Action:
        expressive = self.expressive
        if states.config is not None and "expressive" in states.config:
            expressive = states.config["expressive"]
        if expressive:
            states.expr_vocoder_states.upstream_states = states.upstream_states
            action = self.expr_vocoder.policy(states.expr_vocoder_states)
            if len(states.expr_vocoder_states.source) == 0:
                states.vocoder_states.source = []
        else:
            action = self.vocoder.policy(states.vocoder_states)
            if len(states.vocoder_states.source) == 0:
                states.expr_vocoder_states.source = []
        return action
