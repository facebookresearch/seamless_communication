# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Optional, Sequence, Union
from seamless_communication.padding import get_seqs_and_padding_mask
from typing_extensions import TypeAlias, TypeGuard
from typing import TYPE_CHECKING, List, Optional, overload

import torch
from fairseq2.device import Device
from torch import Tensor
from fairseq2.data.tokenizers import VocabularyInfo

from fairseq2.data import SequenceData
from fairseq2.data.data_pipeline import Collater

class TextTokenizer(ABC):
    """Represents a tokenizer to encode and decode text."""

    vocab_info: VocabularyInfo

    def __init__(self, vocab_info: VocabularyInfo) -> None:
        """
        :param vocab_info:
            The vocabulary information associated with the tokenizer.
        """
        self.vocab_info = vocab_info

    @abstractmethod
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Create a token encoder.

        The valid arguments for the ``task``, ``lang``, and ``mode`` parameters
        are implementation specific. Refer to concrete ``TextTokenizer``
        subclasses for more information.

        :param task:
            The task for which to generate token indices. Typically, multi-task
            jobs use ``task`` to distinguish between different tasks such as
            'translation' or 'transcription'.
        :param lang:
            The language of generated token indices. Typically, multilingual
            translation tasks use ``lang`` to distinguish between different
            languages such as 'en-US' or 'de-DE'.
        :param mode:
            The mode in which to generate token indices. Typically, translation
            tasks use ``mode`` to distinguish between different modes such as
            'source' or 'target'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        """Create a raw token encoder with no control symbols.

        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """

    @abstractmethod
    def create_decoder(self) -> TextTokenDecoder:
        """Create a token decoder."""
class TextTokenEncoder(ABC):
    """Encodes text into tokens or token indices."""

    @abstractmethod
    def __call__(self, text) -> Tensor:
        """
        :param text:
            The text to encode.
        """

    @abstractmethod
    def encode_as_tokens(self, text):
        """
        :param text:
            The text to encode.
        """

    @property
    @abstractmethod
    def prefix_indices(self) -> Optional[Tensor]:
        """Get the indices of the prefix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""

    @property
    @abstractmethod
    def suffix_indices(self) -> Optional[Tensor]:
        """Get the indices of the suffix tokens. *Shape:* :math:`(S)`, where
        :math:`S` is the number of indices."""

class TextTokenDecoder(ABC):
    """Decodes text from tokens or token indices."""

    @abstractmethod
    def __call__(self, token_indices: Tensor):
        """
        :param token_indices:
            The token indices to decode from.
        """

    @abstractmethod
    def decode_from_tokens(self, tokens):
        """
        :param tokens:
            The tokens to decode from.
        """

from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig

from seamless_communication.models.unity.model import UnitYModel
from simuleval.agents import SpeechToSpeechAgent
from simuleval.agents.actions import Action, ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment
from seamless_communication.streaming.agents.common import (
    AgentStates,
    NoUpdateTargetMixin,
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
        super().__init__(args)
        self.model = unity_model
        self.w2v2_encoder_config = w2v2_encoder_config
        self.collate = Collater(
            pad_value=text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )
        self.device = args.device
        self.dtype = args.dtype
        self.min_starting_wait = args.min_starting_wait_w2vbert

    @property
    def min_input_length(self) -> int:
        return self.w2v2_encoder_config.fbank_stride

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
        import time
        print("Wav2Vec", time.time())
        if (
            self.min_starting_wait is not None
            and len(states.source) < self.min_starting_wait
            and not states.source_finished
        ):
            return ReadAction()

        if len(states.source) < self.min_input_length:
            if states.source_finished:
                return WriteAction({}, finished=states.source_finished)
            else:
                return ReadAction()

        inputs = torch.stack(states.source).to(device=self.device, dtype=self.dtype)
        src: SequenceData = self.collate(inputs)

        seqs, padding_mask = get_seqs_and_padding_mask(src)
        encoder_output, _ = self.model.encode_speech(
            seqs,
            padding_mask,
        )

        return WriteAction(
            SpeechSegment(
                content=encoder_output,
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
