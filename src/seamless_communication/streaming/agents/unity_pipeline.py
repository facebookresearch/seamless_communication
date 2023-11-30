# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Union

import torch
from fairseq2.assets import asset_store
from seamless_communication.inference.translator import Modality, Translator
from seamless_communication.models.generator.loader import load_pretssel_vocoder_model
from seamless_communication.models.generator.vocoder import PretsselVocoder
from seamless_communication.models.monotonic_decoder import (
    load_monotonic_decoder_config,
    load_monotonic_decoder_model,
)
from seamless_communication.models.unity import (
    load_unity_config,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.vocoder.loader import load_vocoder_model
from seamless_communication.models.vocoder.vocoder import Vocoder
from seamless_communication.streaming.agents.common import (
    AgentStates,
    EarlyStoppingMixin,
)
from simuleval.agents import AgentPipeline, TreeAgentPipeline
from simuleval.agents.agent import GenericAgent
from simuleval.data.segments import Segment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def maybe_reset_states(states: Optional[List[Optional[AgentStates]]]) -> None:
    assert states is not None
    for s in states:
        if s is not None:
            if isinstance(s, EarlyStoppingMixin):
                s.reset_early()
            else:
                s.reset()


class UnitYPipelineMixin:
    """
    Mixin for UnitY pipeline which works with both AgentPipeline
    and TreeAgentPipeline
    """

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        super().add_args(parser)  # type: ignore
        parser.add_argument("--task", type=str, help="Task type")
        parser.add_argument(
            "--unity-model-name",
            type=str,
            help="Unity model name.",
            default="seamless_streaming_unity",
        )
        parser.add_argument(
            "--monotonic-decoder-model-name",
            type=str,
            help="Monotonic decoder model name.",
            default="seamless_streaming_monotonic_decoder",
        )

        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=float,
        )
        parser.add_argument(
            "--dtype",
            choices=["fp16", "fp32"],
            default="fp16",
            type=str,
            help=(
                "Choose between half-precision (fp16) and single precision (fp32) floating point formats."
                + " Prefer this over the fp16 flag."
            ),
        )

    @classmethod
    def load_model(cls, args: Namespace) -> Dict[str, Any]:
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

        monotonic_decoder_config = load_monotonic_decoder_config(
            args.monotonic_decoder_model_name
        )
        logger.info(
            f"Loading the Monotonic Decoder model: {args.monotonic_decoder_model_name} on device={args.device}, dtype={args.dtype}"
        )
        monotonic_decoder_model = load_monotonic_decoder_model(
            args.monotonic_decoder_model_name, device=args.device, dtype=args.dtype
        )
        monotonic_decoder_model.eval()

        return {
            "unity_model": unity_model,
            "unity_config": unity_config,
            "monotonic_decoder_model": monotonic_decoder_model,
            "monotonic_decoder_config": monotonic_decoder_config,
            "text_tokenizer": text_tokenizer,
            "unit_tokenizer": unit_tokenizer,
        }


class UnitYAgentPipeline(UnitYPipelineMixin, AgentPipeline):  # type: ignore
    pipeline: List[GenericAgent] = []

    def __init__(self, args: Namespace):
        models_and_configs = self.load_model(args)

        module_list = []
        for p in self.pipeline:
            module_list.append(
                p.from_args(
                    args,
                    **models_and_configs,
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

    @classmethod
    def from_args(cls, args: Any) -> UnitYAgentPipeline:
        return cls(args)


class UnitYAgentTreePipeline(UnitYPipelineMixin, TreeAgentPipeline):  # type: ignore
    pipeline: Any = {}

    def __init__(self, args: Namespace):
        models_and_configs = self.load_model(args)

        assert len(self.pipeline) > 0
        module_dict = {}
        for module_class, children in self.pipeline.items():
            module_dict[module_class.from_args(args, **models_and_configs)] = children

        super().__init__(module_dict, args)

    @classmethod
    def from_args(cls, args: Any) -> UnitYAgentPipeline:
        return cls(args)

    def pop(
        self, states: Optional[List[Optional[AgentStates]]] = None
    ) -> List[Segment]:
        output_segment = super().pop(states)
        if states is None:
            # Not stateless
            first_states = self.source_module.states
        else:
            assert len(states) == len(self.module_dict)
            first_states = states[self.source_module]

        if isinstance(output_segment, list):
            finished = any(segment.finished for segment in output_segment)
        else:
            # case when output_index is used
            finished = output_segment.finished
        if not first_states.source_finished and finished:
            # An early stop.
            # The temporary solution is to start over
            if states is not None:
                maybe_reset_states(states)
            else:
                self.reset()
            if isinstance(output_segment, list):
                for segment in output_segment:
                    segment.finished = False
            else:
                output_segment.finished = False

        return output_segment  # type: ignore[no-any-return]
