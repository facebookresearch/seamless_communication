# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, final

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils.model_loader import ModelLoader
from overrides import override as finaloverride

from seamless_communication.models.vocoder.builder import (
    VocoderConfig,
    create_mel_vocoder_model,
    create_vocoder_model,
    mel_vocoder_archs,
    vocoder_archs,
)
from seamless_communication.models.vocoder.melhifigan import MelGenerator
from seamless_communication.models.vocoder.vocoder import Vocoder


@final
class VocoderLoader(ModelLoader[Vocoder, VocoderConfig]):
    """Loads Vocoder models."""

    @finaloverride
    def _convert_checkpoint(
        self, checkpoint: Mapping[str, Any], config: VocoderConfig
    ) -> Mapping[str, Any]:
        old_state_dict = checkpoint["generator"]
        new_state_dict = {}
        for key in old_state_dict:
            new_key = f"code_generator.{key}"
            new_state_dict[new_key] = old_state_dict[key]
        checkpoint["model"] = new_state_dict
        del checkpoint["generator"]  # type: ignore
        return checkpoint


load_vocoder_model = VocoderLoader(
    asset_store, download_manager, create_vocoder_model, vocoder_archs
)


@final
class MelVocoderLoader(ModelLoader[MelGenerator, VocoderConfig]):
    @finaloverride
    def _convert_checkpoint(
        self, checkpoint: Mapping[str, Any], config: VocoderConfig
    ) -> Mapping[str, Any]:
        return checkpoint


load_mel_vocoder_model = MelVocoderLoader(
    asset_store, download_manager, create_mel_vocoder_model, mel_vocoder_archs
)
