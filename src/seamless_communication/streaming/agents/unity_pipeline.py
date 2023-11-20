# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
from simuleval.agents.agent import GenericAgent

import logging
import torch

from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional

from fairseq2.assets import asset_store
from seamless_communication.streaming.agents.mixins import EarlyStoppingMixin
from seamless_communication.inference.translator import Modality, Translator
from seamless_communication.models.unity import (
    load_unity_config,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.monotonic_decoder import load_monotonic_decoder_model

from simuleval.agents import AgentPipeline, AgentStates
from simuleval.data.segments import Segment


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def maybe_reset_states(states: Optional[List[Optional[AgentStates]]]) -> None:
    for s in states:
        if s is not None:
            if isinstance(s, EarlyStoppingMixin):
                s.reset_early()
            else:
                s.reset()


class UnitYPipelineMixin:
    """
    Mixin for fairseq pipeline which works with both AgentPipeline
    and TreeAgentPipeline
    """

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        super().add_args(parser)
        parser.add_argument("--task", type=str, help="Task type")
        parser.add_argument(
            "--unity-model-name",
            type=str,
            help="Unity model name.",
            default="unity_sans_decoder",
        )
        parser.add_argument(
            "--monotonic-decoder-model-name",
            type=str,
            help="Monotonic decoder model name.",
            default="monotonic_decoder",
        )
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=float,
        )
        parser.add_argument(
            "--dtype",
            default="fp16",
            type=str,
        )

    @classmethod
    def from_args(cls, args: Any) -> UnitYPipelineMixin:
        return cls(args)


class UnitYAgentPipeline(UnitYPipelineMixin, AgentPipeline):
    pipeline: List[GenericAgent] = []

    def __init__(self, args: Namespace):

        if not torch.cuda.is_available() and "cuda" in args.device:
            raise ValueError("CUDA not available, use CPU.")

        args.device = torch.device(args.device)
        if (args.fp16 or args.dtype == "fp16") and args.device != torch.device("cpu"):
            args.dtype = torch.float16
        else:
            args.dtype = torch.float32

        input_modality, output_modality = Translator.get_modalities_from_task_str(
            args.task
        )

        if input_modality != Modality.SPEECH:
            raise ValueError("`UnitYAgentPipeline` only supports speech input.")

        unity_config = load_unity_config(args.unity_model_name)
        unity_config.use_text_decoder = False
        unity_config.use_text_encoder = False

        text_tokenizer = load_unity_text_tokenizer(args.unity_model_name)

        # Skip loading the T2U model.
        if output_modality == Modality.TEXT:
            unity_config.t2u_config = None
            unit_tokenizer = None
        else:
            unit_tokenizer = load_unity_unit_tokenizer(args.unity_model_name)

        asset_card = asset_store.retrieve_card(args.unity_model_name)
        asset_card.field("model_config").set(unity_config)

        logger.info(
            f"Loading the UnitY model: {args.unity_model_name} on device={args.device}, dtype={args.dtype}"
        )
        unity_model = load_unity_model(asset_card, device=args.device, dtype=args.dtype)
        unity_model.eval()

        logger.info(
            f"Loading the Monotonic Decoder model: {args.monotonic_decoder_model_name} on device={args.device}, dtype={args.dtype}"
        )
        monotonic_decoder_model = load_monotonic_decoder_model(
            args.monotonic_decoder_model_name, device=args.device, dtype=args.dtype
        )
        monotonic_decoder_model.eval()

        module_list = []
        for p in self.pipeline:
            module_list.append(
                p.from_args(
                    args,
                    unity_model=unity_model,
                    unity_config=unity_config,
                    monotonic_decoder_model=monotonic_decoder_model,
                    text_tokenizer=text_tokenizer,
                    unit_tokenizer=unit_tokenizer,
                )
            )

        super().__init__(module_list)

    def pop(self, states: Optional[List[Optional[AgentStates]]] = None) -> Segment:
        output_segment = super().pop(states)
        if states is None:
            # Not stateless
            first_states = self.module_list[0].states
        else:
            assert len(states) == len(self.module_list)
            first_states = states[0]

        if not first_states.source_finished and output_segment.finished:
            # An early stop.
            # The temporary solution is to start over
            if states is not None:
                maybe_reset_states(states)
            else:
                self.reset()
            output_segment.finished = False

        return output_segment
