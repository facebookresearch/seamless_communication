# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


from typing import Any, Mapping

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader

from seamless_communication.models.generator.builder import (
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
)
from seamless_communication.models.generator.vocoder import PretsselVocoder

load_pretssel_vocoder_config = ConfigLoader[VocoderConfig](asset_store, vocoder_archs)


load_pretssel_vocoder_model = ModelLoader[PretsselVocoder, VocoderConfig](
    asset_store,
    download_manager,
    load_pretssel_vocoder_config,
    create_vocoder_model,
    restrict_checkpoints=False,
)
