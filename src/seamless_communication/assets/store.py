# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from fairseq2.assets import AssetStore
from fairseq2.assets.card_storage import LocalAssetCardStorage
from fairseq2.assets.store import DefaultAssetStore


def create_default_asset_store() -> AssetStore:
    pathname = Path(__file__).parent.joinpath("cards")

    card_storage = LocalAssetCardStorage(pathname)

    return DefaultAssetStore(card_storage)


asset_store = create_default_asset_store()
