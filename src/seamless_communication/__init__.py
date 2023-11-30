# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path

from fairseq2.assets import FileAssetMetadataProvider, asset_store

__version__ = "0.1.0"


def _update_asset_store() -> None:
    cards_dir = Path(__file__).parent.joinpath("cards")

    asset_store.metadata_providers.append(FileAssetMetadataProvider(cards_dir))


_update_asset_store()
