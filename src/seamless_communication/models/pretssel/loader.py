# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, final

from fairseq2.assets import asset_store, download_manager
from fairseq2.models.utils.checkpoint_loader import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelLoader
from overrides import override as finaloverride

from seamless_communication.models.pretssel.builder import (
    PretsselConfig,
    create_pretssel_model,
    pretssel_archs,
)
from seamless_communication.models.pretssel.pretssel_model import PretsselModel


@final
class PretsselLoader(ModelLoader[PretsselModel, PretsselConfig]):
    """Load PretsselModel."""

    @finaloverride
    def _convert_checkpoint(
        self, checkpoint: Mapping[str, Any], config: PretsselConfig
    ) -> Mapping[str, Any]:
        state_dict = checkpoint["model"]

        # Check if we have a fairseq2 checkpoint.
        if "decoder_frontend.embed.weight" in state_dict:
            return checkpoint

        key_map = self._fairseq_key_map(config)

        checkpoint = upgrade_fairseq_checkpoint(checkpoint, key_map)

        state_dict = checkpoint["model"]

        keys_to_delete = []

        keys_to_delete.extend(
            [
                "encoder.embed_positions._float_tensor",
                "decoder.embed_positions._float_tensor",
                "enc_emb_proj.weight",
                "enc_emb_proj.bias",
            ]
        )

        keys_to_delete.extend(
            [
                key
                for key in state_dict
                if key.startswith("decoder.var_adaptor.duration_predictor")
            ]
        )

        for key in keys_to_delete:
            if key in state_dict:
                del state_dict[key]

        return checkpoint

    @staticmethod
    def _fairseq_key_map(config: PretsselConfig) -> Dict[str, str]:
        key_map = {
            # fmt: off
            # encoder frontend
            r"^prosody_encoder\.":                                        r"encoder_frontend.prosody_encoder.",
            r"^encoder\.embed_tokens\.":                                  r"encoder_frontend.embed_tokens.",
            r"^embed_lang\.":                                             r"encoder_frontend.embed_lang.",
            r"^encoder\.pos_emb_alpha":                                   r"encoder_frontend.pos_emb_alpha",

            # encoder
            r"^encoder\.fft_layers\.([0-9]+)\.self_attn\.out_proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
            r"^encoder\.fft_layers\.([0-9]+)\.self_attn\.":               r"encoder.layers.\1.self_attn.",
            r"^encoder\.fft_layers\.([0-9]+)\.layer_norm\.":              r"encoder.layers.\1.self_attn_layer_norm.",
            r"^encoder\.fft_layers\.([0-9]+)\.ffn\.ffn\.0\.":             r"encoder.layers.\1.conv1d.conv1.",
            r"^encoder\.fft_layers\.([0-9]+)\.ffn\.ffn\.2\.":             r"encoder.layers.\1.conv1d.conv2.",
            r"^encoder\.fft_layers\.([0-9]+)\.ffn\.layer_norm\.":         r"encoder.layers.\1.conv1d_layer_norm.",
            r"^encoder\.fft_layers\.([0-9]+)\.film\.":                    r"encoder.layers.\1.film.",

            # decoder frontend
            r"^decoder\.var_adaptor\.":                                   r"decoder_frontend.variance_adaptor.",
            r"^decoder\.pos_emb_alpha":                                   r"decoder_frontend.pos_emb_alpha",

            # decoder
            r"^decoder\.fft_layers\.([0-9]+)\.self_attn\.out_proj\.":     r"decoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.fft_layers\.([0-9]+)\.self_attn\.":               r"decoder.layers.\1.self_attn.",
            r"^decoder\.fft_layers\.([0-9]+)\.layer_norm\.":              r"decoder.layers.\1.self_attn_layer_norm.",
            r"^decoder\.fft_layers\.([0-9]+)\.ffn\.ffn\.0\.":             r"decoder.layers.\1.conv1d.conv1.",
            r"^decoder\.fft_layers\.([0-9]+)\.ffn\.ffn\.2\.":             r"decoder.layers.\1.conv1d.conv2.",
            r"^decoder\.fft_layers\.([0-9]+)\.ffn\.layer_norm\.":         r"decoder.layers.\1.conv1d_layer_norm.",
            r"^decoder\.fft_layers\.([0-9]+)\.film\.":                    r"decoder.layers.\1.film.",

            # final_proj & postnet
            r"^decoder\.out_proj\.":                                      r"final_proj.",
            r"^decoder\.postnet\.":                                       r"postnet.",
            # fmt: on
        }

        return key_map


load_pretssel_model = PretsselLoader(
    asset_store,
    download_manager,
    create_pretssel_model,
    pretssel_archs,
    restrict_checkpoints=False,
)
