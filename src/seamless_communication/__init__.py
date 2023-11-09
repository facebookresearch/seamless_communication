# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from fairseq2.assets import LocalAssetCardStorage, asset_store

__version__ = "0.1.0"


def _update_asset_store() -> None:
    pathname = Path(__file__).parent.joinpath("cards")

    card_storage = LocalAssetCardStorage(pathname)

    asset_store.add_storage(card_storage)


_update_asset_store()
