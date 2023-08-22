# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Mapping, Union, final

import torch
from fairseq2.assets import AssetStore, download_manager
from fairseq2.assets.card import AssetCard
from fairseq2.models.nllb.loader import NllbTokenizerLoader
from seamless_communication.models.unity.builder import (
    UnitYConfig,
    create_unity_model,
    unity_archs,
)
from seamless_communication.models.unity.model import UnitYModel
from seamless_communication.models.unity.unit_tokenizer import UnitTokenizer
from fairseq2.models.utils.checkpoint_loader import upgrade_fairseq_checkpoint
from fairseq2.models.utils.model_loader import ModelConfigLoader, ModelLoader
from overrides import override as finaloverride

from seamless_communication.assets import asset_store


@final
class UnitYLoader(ModelLoader[UnitYModel, UnitYConfig]):
    """Loads UnitY models."""

    @finaloverride
    def _upgrade_checkpoint(
        self, checkpoint: Mapping[str, Any], config: UnitYConfig
    ) -> Mapping[str, Any]:
        state_dict = checkpoint["model"]

        # Check if we have a fairseq2 checkpoint.
        if "decoder_frontend.embed.weight" in state_dict:
            return checkpoint

        key_map = self._fairseq_key_map(config)

        checkpoint = upgrade_fairseq_checkpoint(checkpoint, key_map)

        state_dict = checkpoint["model"]

        # Use the built-in version attribute of `torch.Module`.
        del state_dict["target_letter_decoder.version"]
        del state_dict["target_letter_decoder.embed_positions._float_tensor"]

        if config.use_text_encoder:
            if "text_encoder.version" in state_dict:
                del state_dict["text_encoder.version"]
            if "text_encoder.embed_positions._float_tensor" in state_dict:
                del state_dict["text_encoder.embed_positions._float_tensor"]

        # Remnant of wav2vec2 pretraining, not needed for eval or fine-tuning.
        del state_dict["encoder.w2v_encoder.w2v_model.mask_emb"]

        embeds = state_dict["final_proj.weight"]

        # fairseq had a bug that accidentally introduced a dummy token in the
        # embedding table of NLLB-100. We just discard it.
        if embeds.size(0) == 256103:  # means NLLB-100
            embeds = embeds[:-1]

            state_dict["final_proj.weight"] = embeds

        # fairseq checkpoints have duplicate embedding weights. Ensure that we
        # use a single embedding table in fairseq2.
        state_dict["text_decoder_frontend.embed.weight"] = embeds

        if config.use_text_encoder:
            state_dict["text_encoder_frontend.embed.weight"] = embeds

        # The embedding positions of the control symbols in fairseq's dict do
        # not match the SentencePiece model of the tokenizer.
        with torch.inference_mode():
            # (BOS, PAD, EOS, UNK) -> (PAD, UNK, BOS, EOS)
            embeds[[0, 1, 2, 3]] = embeds[[1, 3, 0, 2]]

        if config.t2u_config is not None:
            # fairseq checkpoints have duplicate embedding weights. Ensure that we
            # use a single embedding table in fairseq2.
            embeds = state_dict["t2u_model.final_proj.weight"]

            state_dict["t2u_model.decoder_frontend.embed.weight"] = embeds

        return checkpoint

    @staticmethod
    def _fairseq_key_map(config: UnitYConfig) -> Dict[str, str]:
        key_map = {
            # fmt: off

            # Speech Encoder
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.pos_conv\.0\.":                                    r"speech_encoder_frontend.pos_encoder.conv.",
            r"^encoder\.w2v_encoder\.w2v_model\.layer_norm\.":                                              r"speech_encoder_frontend.post_extract_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.post_extract_proj\.":                                       r"speech_encoder_frontend.model_dim_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"speech_encoder.inner.layers.\1.conv.batch_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"speech_encoder.inner.layers.\1.conv.depthwise_conv.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"speech_encoder.inner.layers.\1.conv_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"speech_encoder.inner.layers.\1.conv.pointwise_conv1.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"speech_encoder.inner.layers.\1.conv.pointwise_conv2.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"speech_encoder.inner.layers.\1.ffn\2_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"speech_encoder.inner.layers.\1.ffn\2.inner_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"speech_encoder.inner.layers.\1.ffn\2.output_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"speech_encoder.inner.layers.\1.self_attn_layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"speech_encoder.inner.layers.\1.self_attn.q_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"speech_encoder.inner.layers.\1.self_attn.k_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"speech_encoder.inner.layers.\1.self_attn.v_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"speech_encoder.inner.layers.\1.self_attn.output_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"speech_encoder.inner.layers.\1.self_attn.sdpa.r_proj.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"speech_encoder.inner.layers.\1.self_attn.sdpa.u_bias",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"speech_encoder.inner.layers.\1.self_attn.sdpa.v_bias",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"speech_encoder.inner.layers.\1.layer_norm.",
            r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.":                                     r"speech_encoder.inner.layer_norm.",

            # Speech Encoder Adaptor
            r"^encoder\.adaptor\.proj\.0\.": r"speech_encoder.proj1.",
            r"^encoder\.adaptor\.proj\.2\.": r"speech_encoder.proj2.",
            r"^encoder\.adaptor\.out_ln\.":  r"speech_encoder.layer_norm.",

            # Text Encoder
            r"^text_encoder\.embed_tokens\.":                              r"text_encoder_frontend.embed.",
            r"^text_encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"text_encoder.layers.\1.self_attn.output_proj.",
            r"^text_encoder\.layers\.([0-9]+)\.self_attn\.":               r"text_encoder.layers.\1.self_attn.",
            r"^text_encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"text_encoder.layers.\1.self_attn_layer_norm.",
            r"^text_encoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"text_encoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^text_encoder\.layers\.([0-9]+)\.encoder_attn\.":            r"text_encoder.layers.\1.encoder_decoder_attn.",
            r"^text_encoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"text_encoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^text_encoder\.layers\.([0-9]+)\.fc1\.":                     r"text_encoder.layers.\1.ffn.inner_proj.",
            r"^text_encoder\.layers\.([0-9]+)\.fc2\.":                     r"text_encoder.layers.\1.ffn.output_proj.",
            r"^text_encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"text_encoder.layers.\1.ffn_layer_norm.",
            r"^text_encoder\.layer_norm\.":                                r"text_encoder.layer_norm.",

            # Text Decoder
            r"^target_letter_decoder\.embed_tokens\.":                              r"text_decoder_frontend.embed.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"text_decoder.layers.\1.self_attn.output_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.self_attn\.":               r"text_decoder.layers.\1.self_attn.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"text_decoder.layers.\1.self_attn_layer_norm.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"text_decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"text_decoder.layers.\1.encoder_decoder_attn.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"text_decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.fc1\.":                     r"text_decoder.layers.\1.ffn.inner_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.fc2\.":                     r"text_decoder.layers.\1.ffn.output_proj.",
            r"^target_letter_decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"text_decoder.layers.\1.ffn_layer_norm.",
            r"^target_letter_decoder\.layer_norm\.":                                r"text_decoder.layer_norm.",
            r"^target_letter_decoder\.output_projection\.":                         r"final_proj.",

            # T2U Encoder
            r"^synthesizer_encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"t2u_model.encoder.layers.\1.self_attn.output_proj.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.self_attn\.":               r"t2u_model.encoder.layers.\1.self_attn.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"t2u_model.encoder.layers.\1.self_attn_layer_norm.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.fc1\.":                     r"t2u_model.encoder.layers.\1.ffn.inner_proj.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.fc2\.":                     r"t2u_model.encoder.layers.\1.ffn.output_proj.",
            r"^synthesizer_encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"t2u_model.encoder.layers.\1.ffn_layer_norm.",
            r"^synthesizer_encoder\.layer_norm\.":                                r"t2u_model.encoder.layer_norm.",

            # T2U Decoder
            r"^decoder\.embed_tokens\.":                              r"t2u_model.decoder_frontend.embed.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"t2u_model.decoder.layers.\1.self_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.self_attn\.":               r"t2u_model.decoder.layers.\1.self_attn.",
            r"^decoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"t2u_model.decoder.layers.\1.self_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"t2u_model.decoder.layers.\1.encoder_decoder_attn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"t2u_model.decoder.layers.\1.encoder_decoder_attn.",
            r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"t2u_model.decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"t2u_model.decoder.layers.\1.ffn.inner_proj.",
            r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"t2u_model.decoder.layers.\1.ffn.output_proj.",
            r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"t2u_model.decoder.layers.\1.ffn_layer_norm.",
            r"^decoder\.layer_norm\.":                                r"t2u_model.decoder.layer_norm.",
            r"^decoder\.output_projection\.":                         r"t2u_model.final_proj.",
            # fmt: on
        }

        # In normal circumstances, we should never encounter a `LayerNorm` when
        # `use_conformer` is `True`. Unfortunately, the w2v-BERT pretraining in
        # fairseq was accidentally run with a pre-LN encoder, and ended up with
        # a redundant `LayerNorm` right after the Conformer blocks. We mitigate
        # that issue here by moving that `LayerNorm` to the adaptor block.
        if config.w2v2_encoder_config.use_conformer:
            key_map.update(
                {
                    r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.": r"speech_encoder.inner_layer_norm."
                }
            )
        else:
            key_map.update(
                {
                    r"^encoder\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.": r"speech_encoder.inner.layer_norm."
                }
            )

        # fmt: off
        if config.use_conformer_adaptor:
            key_map.update(
                {
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":          r"speech_encoder.adaptor_layers.\1.block.self_attn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.":                    r"speech_encoder.adaptor_layers.\1.block.self_attn.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"speech_encoder.adaptor_layers.\1.block.self_attn_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"speech_encoder.adaptor_layers.\1.block.ffn\2_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"speech_encoder.adaptor_layers.\1.block.ffn\2.inner_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"speech_encoder.adaptor_layers.\1.block.ffn\2.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"speech_encoder.adaptor_layers.\1.block.conv.batch_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"speech_encoder.adaptor_layers.\1.block.conv.depthwise_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"speech_encoder.adaptor_layers.\1.block.conv_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"speech_encoder.adaptor_layers.\1.block.conv.pointwise_conv1.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"speech_encoder.adaptor_layers.\1.block.conv.pointwise_conv2.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":             r"speech_encoder.adaptor_layers.\1.block.layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_ln\.":                      r"speech_encoder.adaptor_layers.\1.layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.conv_pool\.1\.":                 r"speech_encoder.adaptor_layers.\1.conv.",
                }
            )
        else:
            key_map.update(
                {
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.residual_layer_norm\.":  r"speech_encoder.adaptor_layers.\1.residual_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.residual_pool\.1\.":     r"speech_encoder.adaptor_layers.\1.residual_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.attn_pool\.1\.":         r"speech_encoder.adaptor_layers.\1.self_attn_conv.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":  r"speech_encoder.adaptor_layers.\1.self_attn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn\.":            r"speech_encoder.adaptor_layers.\1.self_attn.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.": r"speech_encoder.adaptor_layers.\1.self_attn_layer_norm.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.fc1\.":                  r"speech_encoder.adaptor_layers.\1.ffn.inner_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.fc2\.":                  r"speech_encoder.adaptor_layers.\1.ffn.output_proj.",
                    r"^encoder\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":     r"speech_encoder.adaptor_layers.\1.ffn_layer_norm.",
                }
            )
        # fmt: on

        return key_map


load_unity_model = UnitYLoader(
    asset_store, download_manager, create_unity_model, unity_archs
)


load_unity_config = ModelConfigLoader[UnitYConfig](asset_store, unity_archs)


load_unity_text_tokenizer = NllbTokenizerLoader(asset_store, download_manager)


class UnitYUnitTokenizerLoader:
    """Loads speech unit tokenizers of UnitY models."""

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        """
        self.asset_store = asset_store

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> UnitTokenizer:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        return UnitTokenizer(
            card.field("num_units").as_(int), card.field("unit_langs").as_list(str)
        )


load_unity_unit_tokenizer = UnitYUnitTokenizerLoader(asset_store)
