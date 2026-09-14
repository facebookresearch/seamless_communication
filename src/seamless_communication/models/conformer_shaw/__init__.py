# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from seamless_communication.models.conformer_shaw.builder import (
    ConformerShawEncoderFactory as ConformerShawEncoderFactory,
)
from seamless_communication.models.conformer_shaw.builder import (
    ConformerShawEncoderConfig as ConformerShawEncoderConfig,
)

from seamless_communication.models.conformer_shaw.builder import (
    create_conformer_shaw_model as create_conformer_shaw_model,
)
from seamless_communication.models.conformer_shaw.loader import (
    load_conformer_shaw_model as load_conformer_shaw_model,
)
# --- hub pour la famille -----------------------------------------------------
from typing import TypeVar, Generic
from fairseq2.assets import get_asset_store
from fairseq2.models.hub import ModelHub
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver
from fairseq2.models.wav2vec2 import Wav2Vec2Model

_FAMILY_NAME = "conformer_shaw"

def get_conformer_shaw_model_hub() -> ModelHub[ConformerShawEncoderConfig, Wav2Vec2Model]:
    """
    Renvoie le hub pour la famille 'conformer_shaw'.
    Suppose que la famille et ses archs ont été enregistrées via init_fairseq2(extras=...).
    """
    resolver = get_dependency_resolver()  # <- remplace get_global_container()
    family = resolver.resolve(ModelFamily, key=_FAMILY_NAME)
    return ModelHub(family, get_asset_store())
