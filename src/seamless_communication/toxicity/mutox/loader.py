# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


# from fairseq2.assets import asset_store, download_manager
# from fairseq2.models.utils import ConfigLoader, ModelLoader
# from seamless_communication.toxicity.mutox.builder import create_mutox_model
from seamless_communication.toxicity.mutox.classifier import (
    MutoxClassifier,
    MutoxConfig,
)

import typing as tp



def convert_mutox_checkpoint(
    checkpoint: tp.Mapping[str, tp.Any], config: MutoxConfig
) -> tp.Mapping[str, tp.Any]:
    new_dict = {}
    for key in checkpoint:
        if key.startswith("model_all."):
            new_dict[key] = checkpoint[key]
    return {"model": new_dict}


from fairseq2.models.hub import ModelHub
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.assets import get_asset_store

_FAMILY = "mutoxmutox_classifier"

def get_mutox_model_hub() -> ModelHub[MutoxClassifier, MutoxConfig]:
    resolver = get_dependency_resolver()
    family = resolver.resolve(ModelFamily, key=_FAMILY)
    return ModelHub(family, get_asset_store())

# load_mutox_config = ConfigLoader[MutoxConfig](asset_store, mutox_archs)

def load_mutox_config(name: str) -> MutoxConfig:
    return get_mutox_model_hub().get_model_config(name)

# load_mutox_model = ModelLoader[MutoxClassifier, MutoxConfig](
#     asset_store,
#     download_manager,
#     load_mutox_config,
#     create_mutox_model,
#     convert_mutox_checkpoint,
# )

def load_mutox_model(
    name: str,
    *,
    device = None,
    dtype = None,
) -> MutoxClassifier:
    return get_mutox_model_hub().load_model(name, device=device, dtype=dtype)