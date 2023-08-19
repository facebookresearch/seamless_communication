# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch
from fairseq2.assets import DefaultAssetDownloadManager


class SCAssetDownloadManager(DefaultAssetDownloadManager):
    @classmethod
    def _get_pathname(cls, uri: str, sub_dir: str) -> Path:
        hub_dir = Path(torch.hub.get_dir()).expanduser()

        hsh = cls._get_uri_hash(uri)

        filename = cls._get_filename(uri)

        return hub_dir.joinpath(
            "seamless_communication", "assets", sub_dir, hsh, filename
        )


download_manager = SCAssetDownloadManager()
