# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from typing import Optional, Union, final

from fairseq2.data.tokenizers import VocabularyInfo
from fairseq2.assets import (
    AssetDownloadManager,
    AssetStore,
    get_asset_store,
    # download_manager,
    get_asset_download_manager,
)
from fairseq2.assets.card import AssetCard
from fairseq2.data.tokenizers.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
)
from fairseq2.device import Device

from torch import Tensor

def vocab_info_from_sentencepiece(model: SentencePieceModel) -> VocabularyInfo:
    """Return the vocabulary information of ``model``."""
    return VocabularyInfo(
        model.vocabulary_size,
        model.unk_idx,
        model.bos_idx,
        model.eos_idx,
        model.pad_idx,
    )
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


# vocab_info_from_sentencepiece,

from fairseq2.device import Device
from overrides import final as finaloverride

@final
class CharTokenizer(TextTokenizer):
    """A character-level tokenizer used during non-autoregressive T2U decoding."""

    model: SentencePieceModel

    def __init__(self, pathname) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        """
        self.model = SentencePieceModel(pathname)

        vocab_info = vocab_info_from_sentencepiece(self.model)

        super().__init__(vocab_info)

    @finaloverride
    def create_encoder(
        self,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> TextTokenEncoder:
        """Creates a character level encoder."""
        return SentencePieceEncoder(
            self.model,
            device=device,
            pin_memory=pin_memory,
        )

    @finaloverride
    def create_raw_encoder(
        self, *, device: Optional[Device] = None, pin_memory: bool = False
    ) -> TextTokenEncoder:
        return SentencePieceEncoder(self.model, device=device, pin_memory=pin_memory)

    @finaloverride
    def create_decoder(self) -> TextTokenDecoder:
        return SentencePieceDecoder(self.model)


class UnitYCharTokenizerLoader:
    """Loads character-level tokenizers of UnitY models."""

    def __init__(
        self, asset_store: AssetStore, download_manager: AssetDownloadManager
    ) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        :param download_manager:
            The download manager to use.
        """
        self.asset_store = asset_store
        self.download_manager = download_manager

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
        force: bool = False,
        progress: bool = True,
    ) -> CharTokenizer:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        uri = card.field("char_tokenizer").as_uri()

        pathname = self.download_manager.download_tokenizer(
            uri, card.name, force=force, progress=progress
        )

        return CharTokenizer(pathname)


def load_unity_char_tokenizer(*args):
    return UnitYCharTokenizerLoader(get_asset_store(), get_asset_download_manager())(*args)
