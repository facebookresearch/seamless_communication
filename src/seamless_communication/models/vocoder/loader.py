# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from fairseq2.models import setup_model_family

from seamless_communication.models.vocoder.builder import (
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
)
from seamless_communication.models.vocoder.vocoder import VOCODER_CODE_HIFIGAN_FAMILY


def convert_vocoder_checkpoint(
    checkpoint: Dict[str, Any], config: VocoderConfig
) -> Dict[str, Any]:
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


load_vocoder_model, load_vocoder_config = setup_model_family(
    VOCODER_CODE_HIFIGAN_FAMILY,
    VocoderConfig,
    create_vocoder_model,
    vocoder_archs,
    convert_vocoder_checkpoint,
)
