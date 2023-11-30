# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Mapping

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils import ConfigLoader, ModelLoader

from seamless_communication.models.vocoder.builder import (
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
)
from seamless_communication.models.vocoder.vocoder import Vocoder


def convert_vocoder_checkpoint(
    checkpoint: Mapping[str, Any], config: VocoderConfig
) -> Mapping[str, Any]:
    if (
        "model" in checkpoint
        and "code_generator.resblocks.0.convs1.0.weight_g" in checkpoint["model"]
    ):
        return checkpoint

    old_state_dict = checkpoint["generator"]
    new_state_dict = {}
    for key in old_state_dict:
        new_key = f"code_generator.{key}"
        new_state_dict[new_key] = old_state_dict[key]
    checkpoint["model"] = new_state_dict  # type: ignore
    del checkpoint["generator"]  # type: ignore
    return checkpoint


load_vocoder_config = ConfigLoader[VocoderConfig](asset_store, vocoder_archs)


load_vocoder_model = ModelLoader[Vocoder, VocoderConfig](
    asset_store,
    download_manager,
    load_vocoder_config,
    create_vocoder_model,
    convert_vocoder_checkpoint,
)
