# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

from fairseq2.assets import InProcAssetMetadataProvider, asset_store


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
