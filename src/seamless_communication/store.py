# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Dict, List, Tuple, Union
from pathlib import Path

from fairseq2.assets import AssetCard, AssetCardFieldNotFoundError, AssetStore, InProcAssetMetadataProvider, asset_store, download_manager
from fairseq2.models.nllb.loader import NllbTokenizerLoader
from seamless_communication.models.unity.unit_tokenizer import UnitTokenizer


def add_gated_assets(model_dir: Path) -> None:
    asset_store.env_resolvers.append(lambda: "gated")

    model_dir = model_dir.resolve()

    gated_metadata = [
        {
            "name": "seamless_expressivity@gated",
            "checkpoint": model_dir.joinpath("m2m_expressive_unity.pt"),
        },
        {
            "name": "vocoder_pretssel@gated",
            "checkpoint": model_dir.joinpath("pretssel_melhifigan_wm.pt"),
        },
        {
            "name": "vocoder_pretssel_16khz@gated",
            "checkpoint": model_dir.joinpath("pretssel_melhifigan_wm-16khz.pt"),
        },
    ]

    asset_store.metadata_providers.append(InProcAssetMetadataProvider(gated_metadata))


load_unity_text_tokenizer = NllbTokenizerLoader(asset_store, download_manager)


class UnitYUnitTokenizerLoader:
    """Loads speech unit tokenizers of UnitY models."""

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        """
        self.asset_store = asset_store

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> UnitTokenizer:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        return UnitTokenizer(
            card.field("num_units").as_(int),
            card.field("unit_langs").as_list(str),
            card.field("model_arch").as_(str),
        )


load_unity_unit_tokenizer = UnitYUnitTokenizerLoader(asset_store)


class GcmvnStatsLoader:
    """Loads GCMVN stats (mean & std) for ProsodyUnitY."""

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        """
        self.asset_store = asset_store

    def __call__(
        self, model_name_or_card: Union[str, AssetCard]
    ) -> Tuple[List[float], List[float]]:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        try:
            gcmvn_stats: Dict[str, List[float]] = card.field("gcmvn_stats").as_(dict)
        except AssetCardFieldNotFoundError:
            model_override = card.field("model_config").as_(dict)
            gcmvn_stats = model_override["gcmvn_stats"]

        return gcmvn_stats["mean"], gcmvn_stats["std"]


load_gcmvn_stats = GcmvnStatsLoader(asset_store)
