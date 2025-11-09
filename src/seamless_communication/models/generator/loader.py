# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


from typing import Any, Mapping, Optional

from fairseq2.assets import get_asset_store, download_manager
# from fairseq2.models.utils import ConfigLoader, ModelLoader

from seamless_communication.models.generator.builder import (
    VocoderConfig,
    create_vocoder_model
)
from seamless_communication.models.generator.vocoder import PretsselVocoder
from seamless_communication.models.vocoder.loader import get_vocoder_model_hub

from fairseq2.device import Device
from fairseq2.data_type import DataType
# load_pretssel_vocoder_config = ConfigLoader[VocoderConfig](get_asset_store(), vocoder_archs)


# load_pretssel_vocoder_model = ModelLoader[PretsselVocoder, VocoderConfig](
#     get_asset_store(),
#     download_manager,
#     load_pretssel_vocoder_config,
#     create_vocoder_model,
#     restrict_checkpoints=False,
# )

def load_pretssel_vocoder_config(name: str) -> VocoderConfig:
    # ↔ ConfigLoader[...](asset_store, vocoder_archs)
    return get_vocoder_model_hub().get_model_config(name)

def load_pretssel_vocoder_model(
    name: str,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> PretsselVocoder:
    # ↔ ModelLoader[...](..., create_vocoder_model, ...)
    return get_vocoder_model_hub().load_model(name, device=device, dtype=dtype)