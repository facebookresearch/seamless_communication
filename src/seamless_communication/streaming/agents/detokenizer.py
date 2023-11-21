# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Dict

from simuleval.agents import TextToTextAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from seamless_communication.streaming.agents.common import (
    AgentStates,
    NoUpdateTargetMixin,
)


class DetokenizerAgent(TextToTextAgent, NoUpdateTargetMixin):
    def __init__(self, args: Namespace):
        super().__init__(args)
        self.detokenize_only = args.detokenize_only

    @classmethod
    def from_args(cls, args: Namespace, **kwargs: Dict[str, Any]) -> DetokenizerAgent:
        return cls(args)

    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--detokenize-only",
            action="store_true",
            default=True,
            help="Run detokenization without waiting for a new token.",
        )

    def policy(self, states: AgentStates) -> Action:
        possible_full_words = self.decode(" ".join([x for x in states.source]))

        if self.detokenize_only and len(states.source) > 0:
            states.source = []
            if len(possible_full_words) == 0 and not states.source_finished:
                return ReadAction()
            else:
                return WriteAction(possible_full_words, states.source_finished)

        if states.source_finished:
            return WriteAction(possible_full_words, True)
        elif len(possible_full_words.split()) > 1:
            full_word = possible_full_words.split()[0]
            states.source = states.source[-1:]
            return WriteAction(full_word, finished=False)
        else:
            return ReadAction()

    def decode(self, x: str) -> str:
        return x.replace(" ", "").replace("\u2581", " ").strip()
