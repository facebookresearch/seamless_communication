# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader
from seamless_communication.toxicity.mutox.builder import create_mutox_model
from seamless_communication.toxicity.mutox.classifier import (
    MutoxClassifier,
    MutoxConfig,
    mutox_archs,
)

import typing as tp


@mutox_archs.decorator("mutox")
def _base_mutox() -> MutoxConfig:
    return MutoxConfig(
        input_size=1024,
    )


def convert_mutox_checkpoint(
    checkpoint: tp.Mapping[str, tp.Any], config: MutoxConfig
) -> tp.Mapping[str, tp.Any]:
    new_dict = {}
    for key in checkpoint:
        if key.startswith("model_all."):
            new_dict[key] = checkpoint[key]
    return {"model": new_dict}


load_mutox_config = ConfigLoader[MutoxConfig](asset_store, mutox_archs)


load_mutox_model = ModelLoader[MutoxClassifier, MutoxConfig](
    asset_store,
    download_manager,
    load_mutox_config,
    create_mutox_model,
    convert_mutox_checkpoint,
)
