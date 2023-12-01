# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.nn.incremental_state import IncrementalStateBag
from seamless_communication.models.monotonic_decoder import (
    MonotonicDecoderConfig,
    MonotonicDecoderModel,
)
from seamless_communication.streaming.agents.common import AgentStates
from simuleval.agents import GenericAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import Segment, TextSegment
from torch import Tensor


class DecoderAgentStates(AgentStates):  # type: ignore
    def reset(self) -> None:
        self.source_len = 0
        self.target_indices: List[int] = []
        self.tgt_lang = None
        self.ngram_block_count = 0
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
            self.source_len = self.source.size(1)


class OnlineTextDecoderAgent(GenericAgent):  # type: ignore
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
        self.no_early_stop = args.no_early_stop

        self.device = args.device
        self.dtype = args.dtype
        self.eos_idx = text_tokenizer.vocab_info.eos_idx

        tgt_lang = getattr(args, "tgt_lang", None)
        assert tgt_lang is not None
        self.token_encoder = text_tokenizer.create_encoder(lang=tgt_lang, mode="target")
        prefix_indices = self.token_encoder.prefix_indices
        assert prefix_indices is not None
        self.prefix_indices: List[int] = prefix_indices.tolist()

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
            default=1,
            help="Minimal starting waiting source steps",
        )
        parser.add_argument(
            "--no-early-stop",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--tgt-lang",
            default="eng",
            type=str,
        )

    def policy(self, states: DecoderAgentStates) -> Action:
        raise NotImplementedError

    def enforce_tgt_lang_in_prefix(self, states: DecoderAgentStates) -> None:
        if states.tgt_lang:
            tgt_lang_tag = f"__{states.tgt_lang}__"
            tgt_lang_tag_idx = self.text_tokenizer.model.token_to_index(tgt_lang_tag)
            self.prefix_indices[-1] = tgt_lang_tag_idx


class MMATextDecoderAgent(OnlineTextDecoderAgent):  # type: ignore
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
        self.block_ngrams = args.block_ngrams
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
        parser.add_argument(
            "--block-ngrams",
            action="store_true",
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
            self.enforce_tgt_lang_in_prefix(states)
            target_input = torch.tensor(
                self.prefix_indices + states.target_indices,
                device=self.device,
                dtype=torch.int64,
            ).unsqueeze(0)
        else:
            target_input = torch.tensor(
                pred_indices[-1:], device=self.device, dtype=torch.int64
            ).unsqueeze(0)

        encoder_output = states.source
        decoder_output, _, p_choose = self.model.decode(
            target_input, None, encoder_output, None, state_bag=self.state_bag
        )

        logits = self.model.project(decoder_output)
        if self.block_ngrams and states.source_finished:
            all_indices = states.target_indices + pred_indices
            blocked_indices = all_indices[-4:]
            logits[:, :, blocked_indices] = float("-inf")

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
        self,
        states: DecoderAgentStates,
        pred_indices: List[int],
        finished: bool,
        decoder_features_out: Optional[Tensor] = None,
    ) -> TextSegment:
        return TextSegment(
            content=" ".join(
                [self.text_tokenizer.model.index_to_token(idx) for idx in pred_indices]
            ),
            finished=finished,
            tgt_lang=states.tgt_lang,
        )

    def get_blocked_ngrams(self, target_indices: List[int]) -> Optional[Set[str]]:
        # TODO: make it configurable and use itertools
        if not self.block_ngrams:
            return None
        blocked_ngrams = set()
        if len(target_indices) >= 4:
            blocked_ngrams.add(str(target_indices[-4:]))
            blocked_ngrams.add(str(target_indices[-4:-2]))
            blocked_ngrams.add(str(target_indices[-4:-1]))
        if len(target_indices) >= 3:
            blocked_ngrams.add(str(target_indices[-3:]))
            blocked_ngrams.add(str(target_indices[-3:-1]))
        if len(target_indices) >= 2:
            blocked_ngrams.add(str(target_indices[-2:]))
        return blocked_ngrams

    def maybe_block_ngrams(
        self,
        states: DecoderAgentStates,
        pred_indices: List[int],
        decoder_features_out: Tensor,
        blocked_ngrams: Optional[Set[str]],
        index: int,
    ) -> Tuple[bool, Tensor]:
        """
        This check is used to force a READ decision when n-gram repeat
        happens before source_finished
        """
        if not self.block_ngrams or states.source_finished:
            return False, decoder_features_out

        assert blocked_ngrams is not None
        all_indices = states.target_indices + pred_indices + [index]
        for n in [3, 2]:  # TODO: make it configurable
            if len(all_indices) >= n and states.ngram_block_count <= 4:
                if str(all_indices[-n:]) in blocked_ngrams:
                    states.ngram_block_count += 1
                    pred_indices[:] = pred_indices[: -(n - 1)]
                    decoder_features_out = decoder_features_out[:, : -(n - 1)]
                    return True, decoder_features_out
                blocked_ngrams.add(str(all_indices[-n:]))
        return False, decoder_features_out

    @torch.inference_mode()
    def policy(self, states: DecoderAgentStates) -> Action:
        if len(states.source) == 0:
            return ReadAction()

        if states.source_len < self.min_starting_wait and not states.source_finished:
            return ReadAction()

        if states.target_finished:
            return WriteAction("", finished=True)

        if len(states.source) == 0:
            return ReadAction()

        self.state_bag = IncrementalStateBag(4096)

        states.source_len = states.source.size(1)

        pred_indices: List[int] = []
        index = None
        prob = None
        finished = False
        blocked_ngrams = self.get_blocked_ngrams(states.target_indices)
        decoder_features_out = None

        while (
            len(states.target_indices + pred_indices) < self.max_len(states)
            and len(pred_indices) < self.max_consecutive_writes
        ):
            index, prob, decoder_features = self.run_decoder(states, pred_indices)

            if decoder_features_out is None:
                decoder_features_out = decoder_features.new(0)
            decoder_features_out = torch.cat(
                [decoder_features_out, decoder_features], dim=1
            )

            if (
                self.no_early_stop
                and not states.source_finished
                and (prob < self.decision_threshold or index == self.eos_idx)
            ):
                if prob == 1.0:
                    pred_indices = []
                break
            block_ngram, decoder_features_out = self.maybe_block_ngrams(
                states, pred_indices, decoder_features_out, blocked_ngrams, index
            )
            if block_ngram:
                break
            if (
                finished
                or index == self.eos_idx
                or len(states.target_indices + pred_indices) > self.max_len(states)
            ):
                finished = True
                break

            if prob < self.decision_threshold and not states.source_finished:
                break

            pred_indices.append(index)
            if self.state_bag.step_nr == 0:
                self.state_bag.increment_step_nr(
                    len(self.prefix_indices + states.target_indices)
                )
            else:
                self.state_bag.increment_step_nr()

        states.target_indices += pred_indices

        if len(pred_indices) > 0 or finished:
            finished = finished or len(
                states.target_indices + pred_indices
            ) > self.max_len(states)
            states.ngram_block_count = 0
            return WriteAction(
                self.postprocess(states, pred_indices, finished, decoder_features_out),
                finished=finished,
            )
        else:
            return ReadAction()


class MMASpeechToTextDecoderAgent(MMATextDecoderAgent):
    source_type = "speech"


@dataclass
class UnitYTextDecoderOutput:
    decoder_features: Tensor
    tokens: List[str]
    target_indices: Optional[Tensor] = None


class UnitYMMATextDecoderAgent(MMASpeechToTextDecoderAgent):
    """
    MMA UnitY text decoder agent which just prepares the decoder
    output for the downstream agent.
    """

    def postprocess(
        self,
        states: DecoderAgentStates,
        pred_indices: List[int],
        finished: bool,
        decoder_features_out: Optional[Tensor] = None,
    ) -> TextSegment:
        tokens: List[str] = [
            self.text_tokenizer.model.index_to_token(idx) for idx in pred_indices
        ]
        assert decoder_features_out is not None
        token_list = self.prefix_indices + states.target_indices
        if (
            len(pred_indices) > 0
            and pred_indices[-1] != self.text_tokenizer.vocab_info.eos_idx
        ):
            # Append "," to make speech smooth
            # TODO: a temporary solution.
            ending_token_index = self.text_tokenizer.model.token_to_index(",")
            token_list.append(ending_token_index)
            self.state_bag.increment_step_nr()

            _, _, decoder_features = self.run_decoder(states, [ending_token_index])
            decoder_features_out = torch.cat(
                [decoder_features_out, decoder_features], dim=1
            )

        target_input = torch.tensor(
            token_list,
            device=self.device,
            dtype=torch.int64,
        ).unsqueeze(0)

        return TextSegment(
            content=UnitYTextDecoderOutput(decoder_features_out, tokens, target_input),
            finished=finished,
            tgt_lang=states.tgt_lang,
        )
