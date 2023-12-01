# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Optional, Union, final

from fairseq2.assets import (
    AssetDownloadManager,
    AssetStore,
    asset_store,
    download_manager,
)
from fairseq2.assets.card import AssetCard
from fairseq2.data.text import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    TextTokenDecoder,
    TextTokenEncoder,
    TextTokenizer,
    vocab_info_from_sentencepiece,
)
from fairseq2.data.typing import PathLike
from fairseq2.typing import Device, finaloverride


@final
class CharTokenizer(TextTokenizer):
    """A character-level tokenizer used during non-autoregressive T2U decoding."""

    model: SentencePieceModel

    def __init__(self, pathname: PathLike) -> None:
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


load_unity_char_tokenizer = UnitYCharTokenizerLoader(asset_store, download_manager)
