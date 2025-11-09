# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple

import torch
from fairseq2.data import SequenceData
from fairseq2.data.data_pipeline import Collater
from fairseq2.data.text import TextTokenizer
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.unity.model import UnitYModel
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import Segment, SpeechSegment
from seamless_communication.streaming.agents.common import (
    AgentStates,
    NoUpdateTargetMixin,
)

CONTEXT_MULTIPLIER = 2


class OfflineWav2VecBertEncoderStates(AgentStates):  # type: ignore
    """
    Maintain buffered FBANK features plus cached encoder outputs so we only
    re-encode freshly arrived frames.
    """

    def __init__(self) -> None:
        super().__init__()
        self.context_size = 0
        self.reset()

    def configure(self, context_size: int) -> None:
        self.context_size = max(context_size, 0)

    def reset(self) -> None:
        super().reset()
        self.pending_segments: List[torch.Tensor] = []
        self.context_window: Optional[torch.Tensor] = None
        self.cached_encoder_out: Optional[torch.Tensor] = None

    def update_source(self, segment: Segment) -> None:
        self.source_finished = segment.finished
        if segment.is_empty:
            return
        assert isinstance(segment.content, torch.Tensor)
        self.pending_segments.append(segment.content)

    @property
    def total_pending_frames(self) -> int:
        return sum(t.size(0) for t in self.pending_segments)

    def prepare_encoder_input(
        self, min_len: int, force: bool
    ) -> Optional[Tuple[torch.Tensor, int, int]]:
        total = self.total_pending_frames
        if not force and total < min_len:
            return None

        pending_tensor = (
            torch.cat(self.pending_segments, dim=0) if self.pending_segments else None
        )
        context_frames = 0 if self.context_window is None else self.context_window.size(0)

        chunks: List[torch.Tensor] = []
        if context_frames > 0 and self.context_window is not None:
            chunks.append(self.context_window)
        if pending_tensor is not None:
            chunks.append(pending_tensor)

        if not chunks:
            return None

        encoder_input = torch.cat(chunks, dim=0)
        new_frames = encoder_input.size(0) - context_frames
        self.pending_segments = []
        return encoder_input, context_frames, new_frames

    def update_context_window(self, encoder_input: torch.Tensor) -> None:
        if self.context_size <= 0:
            self.context_window = None
            return
        context_slice = encoder_input[-self.context_size :, :]
        self.context_window = context_slice.detach().cpu()

    def append_encoder_output(self, new_output: torch.Tensor) -> None:
        if new_output.numel() == 0:
            return
        if self.cached_encoder_out is None:
            self.cached_encoder_out = new_output
        else:
            self.cached_encoder_out = torch.cat(
                [self.cached_encoder_out, new_output], dim=1
            )


class OfflineWav2VecBertEncoderAgent(NoUpdateTargetMixin, SpeechToSpeechAgent):  # type: ignore
    """
    Incremental encoding of an wav2vec encoder output
    It update the whole encoder states every time when there is a new incoming segment.
    """

    def __init__(
        self,
        unity_model: UnitYModel,
        w2v2_encoder_config: Wav2Vec2EncoderConfig,
        text_tokenizer: TextTokenizer,
        args: Namespace,
    ) -> None:
        self.model = unity_model
        self.w2v2_encoder_config = w2v2_encoder_config
        self.context_overlap = (
            CONTEXT_MULTIPLIER * self.w2v2_encoder_config.fbank_stride
        )
        super().__init__(args)
        self.collate = Collater(
            pad_value=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )
        self.device = args.device
        self.dtype = args.dtype
        self.min_starting_wait = args.min_starting_wait_w2vbert

    @property
    def min_input_length(self) -> int:
        return self.w2v2_encoder_config.fbank_stride

    def build_states(self) -> OfflineWav2VecBertEncoderStates:
        states = OfflineWav2VecBertEncoderStates()
        states.configure(self.context_overlap)
        return states

    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--min-starting-wait-w2vbert",
            default=None,
            type=int,
            help="Min starting wait in w2vbert",
        )

    @torch.inference_mode()
    def policy(self, states: AgentStates) -> Action:
        """
        The policy for encoder is always write
        only if the input is too short
        """
        assert isinstance(states, OfflineWav2VecBertEncoderStates)
        import time
        print("Wav2Vec", time.time())
        if (
            self.min_starting_wait is not None
            and states.total_pending_frames < self.min_starting_wait
            and not states.source_finished
        ):
            return ReadAction()

        force_consume = states.source_finished and (
            states.total_pending_frames > 0 or states.cached_encoder_out is None
        )
        prepared = states.prepare_encoder_input(self.min_input_length, force_consume)
        if prepared is None:
            if states.source_finished:
                if states.cached_encoder_out is None:
                    return WriteAction({}, finished=True)
                return WriteAction(
                    SpeechSegment(
                        content=states.cached_encoder_out,
                        tgt_lang=states.tgt_lang,
                        finished=True,
                    ),
                    finished=True,
                )
            return ReadAction()

        encoder_input, context_frames, _new_frames = prepared
        states.update_context_window(encoder_input)
        src: SequenceData = self.collate([encoder_input])

        seqs, padding_mask = get_seqs_and_padding_mask(src)
        seqs = seqs.to(device=self.device, dtype=self.dtype)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device=self.device)
        encoder_output, _ = self.model.encode_speech(
            seqs,
            padding_mask,
        )

        # Remove the portion that only covers the left-context frames.
        context_tokens = 0
        if context_frames > 0:
            context_tokens = context_frames // self.w2v2_encoder_config.fbank_stride
        if context_tokens > 0:
            context_tokens = min(context_tokens, encoder_output.size(1))
            incremental_output = encoder_output[:, context_tokens:, :]
        else:
            incremental_output = encoder_output

        if incremental_output.size(1) == 0 and not states.source_finished:
            return ReadAction()

        states.append_encoder_output(incremental_output)

        if states.cached_encoder_out is None:
            return ReadAction()

        return WriteAction(
            SpeechSegment(
                content=states.cached_encoder_out,
                tgt_lang=states.tgt_lang,
                finished=states.source_finished,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def from_args(
        cls, args: Namespace, **kwargs: Dict[str, Any]
    ) -> OfflineWav2VecBertEncoderAgent:
        unity_model = kwargs.get("unity_model", None)
        assert isinstance(unity_model, UnitYModel)
        unity_config = kwargs.get("unity_config", None)
        assert unity_config is not None
        text_tokenizer = kwargs.get("text_tokenizer", None)
        assert isinstance(text_tokenizer, TextTokenizer)
        return cls(unity_model, unity_config.w2v2_encoder_config, text_tokenizer, args)
