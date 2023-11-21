# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch

from argparse import ArgumentParser, Namespace
from torch import Tensor
from typing import Any, Dict, List, Tuple

from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.nn.incremental_state import IncrementalStateBag
from seamless_communication.models.monotonic_decoder import (
    MonotonicDecoderConfig,
    MonotonicDecoderModel,
)

from simuleval.agents import GenericAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.data.segments import Segment, TextSegment


class DecoderAgentStates(AgentStates):
    def reset(self) -> None:
        self.source_steps = 0
        self.target_indices: List[int] = []
        self.tgt_lang = None
        super().reset()

    def update_source(self, segment: Segment) -> None:
        """
        Update states from input segment
        Additionlly update incremental states

        Args:
            segment (~simuleval.agents.segments.Segment): input segment
        """
        self.source_finished = segment.finished
        if self.tgt_lang is None and segment.tgt_lang is not None:
            self.tgt_lang = segment.tgt_lang
        if not segment.is_empty:
            self.source = segment.content
            if len(self.source) == 0 and segment.finished:
                self.target_finished = True
                return
            self.source_steps = self.source.size(1)


class OnlineTextDecoderAgent(GenericAgent):
    """
    Online text decoder
    """

    target_type = "text"

    def __init__(
        self,
        model: MonotonicDecoderModel,
        config: MonotonicDecoderConfig,
        text_tokenizer: NllbTokenizer,
        args: Namespace,
    ) -> None:
        super().__init__(args)
        self.model = model
        self.config = config
        self.text_tokenizer = text_tokenizer

        self.max_len_a: int = args.max_len_a
        self.max_len_b: int = args.max_len_b
        self.max_consecutive_writes = self.args.max_consecutive_write
        self.min_starting_wait = args.min_starting_wait
        self.min_starting_wait_reset = args.min_starting_wait_reset
        self.no_early_stop = args.no_early_stop

        self.device = args.device
        self.dtype = args.dtype
        self.eos_idx = text_tokenizer.vocab_info.eos_idx
        token_encoder = text_tokenizer.create_encoder(lang=args.tgt_lang, mode="target")
        prefix_tokens = token_encoder.prefix_indices
        assert prefix_tokens is not None
        self.prefix_tokens: List[int] = prefix_tokens.tolist()

    def build_states(self) -> DecoderAgentStates:
        return DecoderAgentStates()

    def max_len(self, states: DecoderAgentStates) -> int:
        return self.max_len_a * int(states.source.size(1)) + self.max_len_b

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--max-len-a",
            type=int,
            default=1,
            help="Max length of predictions, a in ax + b",
        )
        parser.add_argument(
            "--max-len-b",
            type=int,
            default=200,
            help="Max length of predictions, b in ax + b",
        )
        parser.add_argument(
            "--max-consecutive-write",
            type=int,
            default=50,
            help="Max consecutive writes.",
        )
        parser.add_argument(
            "--min-starting-wait",
            type=int,
            default=12,
            help="Minimal starting waiting source steps",
        )
        parser.add_argument(
            "--min-starting-wait-reset",
            type=int,
            default=0,
            help="Minimal starting waiting source steps",
        )
        parser.add_argument(
            "--no-early-stop",
            action="store_true",
            default=True,
        )

    def policy(self, states: DecoderAgentStates) -> Action:
        raise NotImplementedError


class MMATextDecoderAgent(OnlineTextDecoderAgent):
    def __init__(
        self,
        model: MonotonicDecoderModel,
        config: MonotonicDecoderConfig,
        text_tokenizer: NllbTokenizer,
        args: Namespace,
    ) -> None:
        super().__init__(model, config, text_tokenizer, args=args)

        self.num_decoder_layers = self.config.num_decoder_layers

        self.decision_threshold = args.decision_threshold
        self.decision_method = args.decision_method
        self.p_choose_start_layer = args.p_choose_start_layer

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        OnlineTextDecoderAgent.add_args(parser)
        parser.add_argument(
            "--decision-threshold",
            type=float,
            default=0.5,
            help="The threshold to write an output, from 0 to 1. Small values give low latency.",
        )
        parser.add_argument(
            "--decision-method",
            type=str,
            default="min",
            choices=["mean", "min", "median"],
            help="The method to determine the decision. Either average all attention heads, or just pick the smallest one",
        )
        parser.add_argument(
            "--p-choose-start-layer",
            type=int,
            default=0,
            help="Encoder layer from which p_choose should be considered for selection.",
        )

    @classmethod
    def from_args(
        cls, args: Namespace, **kwargs: Dict[str, Any]
    ) -> MMATextDecoderAgent:
        model = kwargs.get("monotonic_decoder_model", None)
        config = kwargs.get("monotonic_decoder_config", None)
        text_tokenizer = kwargs.get("text_tokenizer", None)

        assert isinstance(model, MonotonicDecoderModel)
        assert isinstance(config, MonotonicDecoderConfig)
        assert isinstance(text_tokenizer, NllbTokenizer)

        return cls(
            model=model,
            config=config,
            text_tokenizer=text_tokenizer,
            args=args,
        )

    def run_decoder(
        self, states: DecoderAgentStates, pred_indices: List[int]
    ) -> Tuple[int, float, Tensor]:
        if len(pred_indices) == 0:
            target_input = torch.tensor(
                self.prefix_tokens + states.target_indices,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0)
        else:
            target_input = torch.tensor(
                pred_indices[-1:], device=self.device, dtype=torch.int64
            ).unsqueeze(0)

        states.source_steps = states.source.size(1)
        torch.cuda.empty_cache()

        encoder_output = states.source
        decoder_output, _, p_choose = self.model.decode(
            target_input, None, encoder_output, None, state_bag=self.state_bag
        )

        logits = self.model.project(decoder_output)

        index = int(logits[0, -1].argmax().item())
        _, tgt_len, src_len = p_choose.size()

        p_choose = p_choose.view(self.num_decoder_layers, -1, tgt_len, src_len)

        if self.decision_method == "min":
            prob = p_choose[self.p_choose_start_layer :, :, -1, -1].min().item()
        elif self.decision_method == "mean":
            prob = p_choose[self.p_choose_start_layer :, :, -1, -1].mean().item()
        else:
            prob = p_choose[self.p_choose_start_layer :, :, -1, -1].median().item()

        return index, prob, decoder_output

    def postprocess(
        self, states: DecoderAgentStates, pred_indices: Tensor, finished: bool
    ) -> TextSegment:
        return TextSegment(
            content=" ".join(
                [self.text_tokenizer.model.index_to_token(idx) for idx in pred_indices]
            ),
            finished=finished,
            tgt_lang=states.tgt_lang,
        )

    @torch.inference_mode()
    def policy(self, states: DecoderAgentStates) -> Action:
        if len(states.source) == 0:
            return ReadAction()

        if states.source_steps < self.min_starting_wait and not states.source_finished:
            return ReadAction()

        if states.target_finished:
            return WriteAction("", finished=True)

        if len(states.source) == 0:
            return ReadAction()

        self.state_bag = IncrementalStateBag(4096)

        pred_indices: List[int] = []
        index = None
        prob = None
        finished = False

        while (
            len(states.target_indices + pred_indices) < self.max_len(states)
            and len(pred_indices) < self.max_consecutive_writes
        ):
            index, prob, _ = self.run_decoder(states, pred_indices)

            if (
                self.no_early_stop
                and prob < self.decision_threshold
                and not states.source_finished
            ):
                break
            if (
                self.no_early_stop
                and index == self.eos_idx
                and not states.source_finished
            ):
                if prob == 1.0:
                    pred_indices = []
                if states.source_steps < self.min_starting_wait_reset:
                    pred_indices = []
                    if len(states.target_indices) < 3:
                        states.target_indices = []
                break
            if (
                finished
                or index == self.eos_idx
                or len(states.target_indices + pred_indices) > self.max_len(states)
            ):
                finished = True
                break

            if (
                not self.no_early_stop
                and prob < self.decision_threshold
                and not states.source_finished
            ):
                break

            pred_indices.append(index)
            if self.state_bag.step == 0:
                self.state_bag.increment_step(
                    len(self.prefix_tokens + states.target_indices)
                )
            else:
                self.state_bag.increment_step()

        states.target_indices += pred_indices

        if len(pred_indices) > 0 or finished:
            finished = finished or len(
                states.target_indices + pred_indices
            ) > self.max_len(states)
            return WriteAction(
                self.postprocess(states, torch.tensor(pred_indices), finished),
                finished=finished,
            )
        else:
            return ReadAction()


class MMASpeechToTextDecoderAgent(MMATextDecoderAgent):
    source_type = "speech"
