# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Mapping, Tuple, Union

import torch
from fairseq2.assets import AssetStore, asset_store, download_manager
from fairseq2.assets.card import AssetCard, AssetCardFieldNotFoundError
from fairseq2.models.nllb import NllbConfig
from fairseq2.models.nllb.loader import NllbTokenizerLoader
from fairseq2.models.utils import ConfigLoader, ModelLoader
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

from seamless_communication.models.unity.builder import (
    UnitYConfig,
    create_unity_model,
    unity_archs,
)
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer
from seamless_communication.models.unity.model import UnitYModel
from seamless_communication.models.unity.unit_tokenizer import UnitTokenizer


def convert_unity_checkpoint(
    checkpoint: Mapping[str, Any], config: UnitYConfig
) -> Mapping[str, Any]:
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "speech_encoder.inner.layers.0.self_attn_layer_norm.weight" in state_dict:
        return checkpoint

    key_map = _fairseq_key_map(config)

    checkpoint = convert_fairseq_checkpoint(checkpoint, key_map)

    state_dict = checkpoint["model"]

    keys_to_delete = []

    # ExpressiveUnitY model (from multi_arch codebase)
    if config.prosody_encoder_config is not None:
        encoder_key = "s2t_model.encoder"
        decoder_key = "s2t_model.decoder"
        t2u_decoder_key = "t2s_model.decoder"
    # X2T/S2T + T2U model.
    elif config.t2u_config is not None:
        encoder_key = "encoder"
        decoder_key = "target_letter_decoder"
        t2u_decoder_key = "decoder"
    # X2T model.
    elif config.use_text_encoder:
        encoder_key = "speech_encoder"
        decoder_key = "shared_decoder"
    # S2T model.
    else:
        encoder_key = "encoder"
        decoder_key = "decoder"

    keys_to_delete.append(f"{decoder_key}.version")
    keys_to_delete.append(f"{decoder_key}.embed_positions._float_tensor")

    if config.use_text_encoder:
        keys_to_delete.append("text_encoder.version")
        keys_to_delete.append("text_encoder.embed_positions._float_tensor")

    if not config.use_text_decoder:
        text_decoder_keys = [key for key in state_dict if key.startswith(decoder_key)]
        keys_to_delete.extend(text_decoder_keys)

    # Remnant of wav2vec2 pretraining, not needed for eval or fine-tuning.
    keys_to_delete.append(f"{encoder_key}.w2v_encoder.w2v_model.mask_emb")

    if config.prosody_encoder_config is not None or config.t2u_config is not None:
        keys_to_delete.append(
            f"{t2u_decoder_key}.char_upsampler.embed_positions._float_tensor"
        )
        keys_to_delete.append(
            f"{t2u_decoder_key}.char_upsampler.embed_tokens_char.weight"
        )

        # Delete AlignmentEncoder keys for inference.
        alignment_encoder_keys = [
            key
            for key in state_dict
            if key.startswith(f"{t2u_decoder_key}.alignment_encoder.")
        ]
        keys_to_delete.extend(alignment_encoder_keys)

    # Delete character-level projection for inference.
    keys_to_delete.extend(
        [
            "decoder_target_letter_decoder.proj.weight",
            "decoder_target_letter_decoder.proj.bias",
        ]
    )

    if config.prosody_encoder_config is not None:
        keys_to_delete.extend(
            [
                f"{t2u_decoder_key}.embed_positions._float_tensor",
                "t2s_model.global_proj_dec.weight",
                "t2s_model.global_proj_dec.bias",
                "t2s_model.decoder_target_letter_nllb_spm_decoder.encoder.proj.weight",
                "t2s_model.decoder_target_letter_nllb_spm_decoder.encoder.proj.bias",
            ]
        )

    for key in keys_to_delete:
        if key in state_dict:
            del state_dict[key]

    if config.use_text_decoder:
        embeds = state_dict["final_proj.weight"]

        # fairseq had a bug that accidentally introduced a dummy token in the
        # embedding table of NLLB-100. We just discard it.
        if (
            isinstance(config.mt_model_config, NllbConfig) and embeds.size(0) == 256103
        ):  # means NLLB-100
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

    char_embeds = state_dict.get("t2u_model.decoder_frontend.embed_char.weight", None)
    if char_embeds is not None:
        index_mapping = _get_char_index_mapping(config)
        vocab_size = len(index_mapping)
        char_embeds[torch.arange(vocab_size)] = char_embeds[index_mapping]

    if config.t2u_config is not None:
        # fairseq checkpoints have duplicate embedding weights. Ensure that we
        # use a single embedding table in fairseq2.
        embeds = state_dict["t2u_model.final_proj.weight"]

        if "t2u_model.decoder_frontend.embed.weight" in state_dict:
            state_dict["t2u_model.decoder_frontend.embed.weight"] = embeds

    return checkpoint


def _get_char_index_mapping(config: UnitYConfig) -> List[int]:
    assert config.t2u_config is not None
    assert config.t2u_config.nar_decoder_config is not None
    char_tokenizer = load_unity_char_tokenizer(
        config.t2u_config.nar_decoder_config.model_name_or_card
    )
    spm_order = [
        char_tokenizer.model.index_to_token(i)
        for i in range(char_tokenizer.model.vocabulary_size)
    ][4:]
    spm_to_dict_mapping = {
        ch: idx
        for (idx, ch) in zip(
            range(4, char_tokenizer.model.vocabulary_size),
            sorted(spm_order),
        )
    }
    model_to_dict_mapping = [0, 1, 2, 3] + [spm_to_dict_mapping[ch] for ch in spm_order]
    return model_to_dict_mapping


def _fairseq_key_map(config: UnitYConfig) -> Dict[str, str]:
    # ExpressiveUnitY model (from multi_arch codebase)
    if config.prosody_encoder_config is not None:
        encoder_key = "s2t_model.encoder"
        decoder_key = "s2t_model.decoder"
        t2u_encoder_key = "t2s_model.encoder"
        t2u_decoder_key = "t2s_model.decoder"
        ecapa_tdnn_key = "global_prosody"
    # X2T/S2T + T2U model.
    elif config.t2u_config is not None:
        encoder_key = "encoder"
        decoder_key = "target_letter_decoder"
        t2u_encoder_key = "synthesizer_encoder"
        t2u_decoder_key = "decoder"
    # X2T model.
    elif config.use_text_encoder:
        encoder_key = "speech_encoder"
        decoder_key = "shared_decoder"
    # S2T model.
    else:
        encoder_key = "encoder"
        decoder_key = "decoder"

    key_map = {
        # fmt: off

        # Speech Encoder
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.pos_conv\.0\.":                                    r"speech_encoder_frontend.pos_encoder.conv.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.layer_norm\.":                                              r"speech_encoder_frontend.post_extract_layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.post_extract_proj\.":                                       r"speech_encoder_frontend.model_dim_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.feature_extractor\.conv_layers\.([0-9]+)\.0\.":             r"speech_encoder_frontend.feature_extractor.layers.\1.conv.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.feature_extractor\.conv_layers\.([0-9]+)\.2\.1\.":          r"speech_encoder_frontend.feature_extractor.layers.\1.layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.feature_extractor\.conv_layers\.0\.2\.":                    r"speech_encoder_frontend.feature_extractor.layers.0.group_norm.",

        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"speech_encoder.inner.layers.\1.conv.batch_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.layer_norm2\.":     r"speech_encoder.inner.layers.\1.conv.layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"speech_encoder.inner.layers.\1.conv.depthwise_conv.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"speech_encoder.inner.layers.\1.conv_layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"speech_encoder.inner.layers.\1.conv.pointwise_conv1.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"speech_encoder.inner.layers.\1.conv.pointwise_conv2.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"speech_encoder.inner.layers.\1.ffn\2_layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"speech_encoder.inner.layers.\1.ffn\2.inner_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"speech_encoder.inner.layers.\1.ffn\2.output_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"speech_encoder.inner.layers.\1.self_attn_layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_q\.":          r"speech_encoder.inner.layers.\1.self_attn.q_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_k\.":          r"speech_encoder.inner.layers.\1.self_attn.k_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_v\.":          r"speech_encoder.inner.layers.\1.self_attn.v_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_out\.":        r"speech_encoder.inner.layers.\1.self_attn.output_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.":            r"speech_encoder.inner.layers.\1.self_attn.q_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.":            r"speech_encoder.inner.layers.\1.self_attn.k_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.":            r"speech_encoder.inner.layers.\1.self_attn.v_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.rel_k_embedding\.":   r"speech_encoder.inner.layers.\1.self_attn.sdpa.rel_k_embed.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":          r"speech_encoder.inner.layers.\1.self_attn.output_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.linear_pos\.":        r"speech_encoder.inner.layers.\1.self_attn.sdpa.r_proj.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_u":          r"speech_encoder.inner.layers.\1.self_attn.sdpa.u_bias",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.self_attn\.pos_bias_v":          r"speech_encoder.inner.layers.\1.self_attn.sdpa.v_bias",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layers\.([0-9]+)\.final_layer_norm\.":             r"speech_encoder.inner.layers.\1.layer_norm.",
        fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.":                                     r"speech_encoder.inner.layer_norm.",

        # Speech Encoder Adaptor
        fr"^{encoder_key}\.adaptor\.proj\.0\.": r"speech_encoder.proj1.",
        fr"^{encoder_key}\.adaptor\.proj\.2\.": r"speech_encoder.proj2.",
        fr"^{encoder_key}\.adaptor\.out_ln\.":  r"speech_encoder.layer_norm.",

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
        # fmt: on
    }

    # In normal circumstances, we should never encounter a `LayerNorm` when
    # `use_conformer` is `True`. Unfortunately, the w2v-BERT pretraining in
    # fairseq was accidentally run with a pre-LN encoder, and ended up with
    # a redundant `LayerNorm` right after the Conformer blocks. We mitigate
    # that issue here by moving that `LayerNorm` to the adaptor block.
    # fmt: off
    if config.w2v2_encoder_config.use_conformer:
        key_map.update(
            {
                fr"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.": r"speech_encoder.inner_layer_norm."
            }
        )
    else:
        key_map.update(
            {
                rf"^{encoder_key}\.w2v_encoder\.w2v_model\.encoder\.layer_norm\.": r"speech_encoder.inner.layer_norm."
            }
        )
    # fmt: on

    if config.use_conformer_adaptor:
        key_map.update(
            {
                # fmt: off
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":          r"speech_encoder.adaptor_layers.\1.block.self_attn.output_proj.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.self_attn\.":                    r"speech_encoder.adaptor_layers.\1.block.self_attn.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.":         r"speech_encoder.adaptor_layers.\1.block.self_attn_layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":         r"speech_encoder.adaptor_layers.\1.block.ffn\2_layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                r"speech_encoder.adaptor_layers.\1.block.ffn\2.inner_proj.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                r"speech_encoder.adaptor_layers.\1.block.ffn\2.output_proj.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_module\.batch_norm\.":      r"speech_encoder.adaptor_layers.\1.block.conv.batch_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":  r"speech_encoder.adaptor_layers.\1.block.conv.depthwise_conv.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_module\.layer_norm\.":      r"speech_encoder.adaptor_layers.\1.block.conv_layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.": r"speech_encoder.adaptor_layers.\1.block.conv.pointwise_conv1.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.": r"speech_encoder.adaptor_layers.\1.block.conv.pointwise_conv2.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":             r"speech_encoder.adaptor_layers.\1.block.layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_ln\.":                      r"speech_encoder.adaptor_layers.\1.layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.conv_pool\.1\.":                 r"speech_encoder.adaptor_layers.\1.conv.",
                # fmt: on
            }
        )
    else:
        key_map.update(
            {
                # fmt: off
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.residual_layer_norm\.":  r"speech_encoder.adaptor_layers.\1.residual_layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.residual_pool\.1\.":     r"speech_encoder.adaptor_layers.\1.residual_conv.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.attn_pool\.1\.":         r"speech_encoder.adaptor_layers.\1.self_attn_conv.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.self_attn\.out_proj\.":  r"speech_encoder.adaptor_layers.\1.self_attn.output_proj.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.self_attn\.":            r"speech_encoder.adaptor_layers.\1.self_attn.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.self_attn_layer_norm\.": r"speech_encoder.adaptor_layers.\1.self_attn_layer_norm.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.fc1\.":                  r"speech_encoder.adaptor_layers.\1.ffn.inner_proj.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.fc2\.":                  r"speech_encoder.adaptor_layers.\1.ffn.output_proj.",
                fr"^{encoder_key}\.adaptor\.layers\.([0-9]+)\.final_layer_norm\.":     r"speech_encoder.adaptor_layers.\1.ffn_layer_norm.",
                # fmt: on
            }
        )

    key_map.update(
        {
            # fmt: off
            # Text Decoder
            fr"^{decoder_key}\.embed_tokens\.":                              r"text_decoder_frontend.embed.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"text_decoder.layers.\1.self_attn.output_proj.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.self_attn\.":               r"text_decoder.layers.\1.self_attn.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"text_decoder.layers.\1.self_attn_layer_norm.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"text_decoder.layers.\1.encoder_decoder_attn.output_proj.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.encoder_attn\.":            r"text_decoder.layers.\1.encoder_decoder_attn.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"text_decoder.layers.\1.encoder_decoder_attn_layer_norm.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.fc1\.":                     r"text_decoder.layers.\1.ffn.inner_proj.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.fc2\.":                     r"text_decoder.layers.\1.ffn.output_proj.",
            fr"^{decoder_key}\.layers\.([0-9]+)\.final_layer_norm\.":        r"text_decoder.layers.\1.ffn_layer_norm.",
            fr"^{decoder_key}\.layer_norm\.":                                r"text_decoder.layer_norm.",
            fr"^{decoder_key}\.output_projection\.":                         r"final_proj.",
            # fmt: on
        }
    )
    # ExpressiveUnitY model (from multi_arch codebase)
    if config.prosody_encoder_config is not None:
        key_map.update(
            {
                # fmt: off
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.film\.":              r"t2u_model.decoder.layers.\1.film.",
                fr"^{ecapa_tdnn_key}\.":                                       r"prosody_encoder_model.",
                r"^t2s_model\.global_proj_enc\.":                             r"t2u_model.prosody_proj.",
                # fmt: on
            }
        )

    # X2T/S2T + T2U model.
    if config.t2u_config is not None:
        key_map.update(
            {
                # fmt: off
                # T2U Encoder
                fr"^{t2u_encoder_key}\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"t2u_model.encoder.layers.\1.self_attn.output_proj.",
                fr"^{t2u_encoder_key}\.layers\.([0-9]+)\.self_attn\.":               r"t2u_model.encoder.layers.\1.self_attn.",
                fr"^{t2u_encoder_key}\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"t2u_model.encoder.layers.\1.self_attn_layer_norm.",
                fr"^{t2u_encoder_key}\.layers\.([0-9]+)\.fc1\.":                     r"t2u_model.encoder.layers.\1.ffn.inner_proj.",
                fr"^{t2u_encoder_key}\.layers\.([0-9]+)\.fc2\.":                     r"t2u_model.encoder.layers.\1.ffn.output_proj.",
                fr"^{t2u_encoder_key}\.layers\.([0-9]+)\.final_layer_norm\.":        r"t2u_model.encoder.layers.\1.ffn_layer_norm.",
                fr"^{t2u_encoder_key}\.layer_norm\.":                                r"t2u_model.encoder.layer_norm.",

                # T2U Decoder frontend
                fr"^{t2u_decoder_key}\.embed_tokens_text\.":                           r"t2u_model.decoder_frontend.embed_char.",
                fr"^{t2u_decoder_key}\.embed_tokens_unit\.":                           r"t2u_model.decoder_frontend.embed.",
                fr"^{t2u_decoder_key}\.embed_tokens\.":                                r"t2u_model.decoder_frontend.embed.",
                fr"^{t2u_decoder_key}\.var_adaptor\.duration_predictor\.":             r"t2u_model.decoder_frontend.variance_adaptor.duration_predictor.",
                fr"^{t2u_decoder_key}\.dec_pos_emb_alpha":                             r"t2u_model.decoder_frontend.pos_emb_alpha",
                fr"^{t2u_decoder_key}\.char_upsampler\.pos_emb_alpha":                 r"t2u_model.decoder_frontend.pos_emb_alpha_char",

                # T2U Decoder
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"t2u_model.decoder.layers.\1.self_attn.output_proj.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.self_attn\.":               r"t2u_model.decoder.layers.\1.self_attn.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.self_attn_layer_norm\.":    r"t2u_model.decoder.layers.\1.self_attn_layer_norm.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.layer_norm\.":              r"t2u_model.decoder.layers.\1.self_attn_layer_norm.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"t2u_model.decoder.layers.\1.encoder_decoder_attn.output_proj.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.encoder_attn\.":            r"t2u_model.decoder.layers.\1.encoder_decoder_attn.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"t2u_model.decoder.layers.\1.encoder_decoder_attn_layer_norm.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.fc1\.":                     r"t2u_model.decoder.layers.\1.ffn.inner_proj.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.fc2\.":                     r"t2u_model.decoder.layers.\1.ffn.output_proj.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.final_layer_norm\.":        r"t2u_model.decoder.layers.\1.ffn_layer_norm.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.ffn\.ffn\.0\.":             r"t2u_model.decoder.layers.\1.conv1d.conv1.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.ffn\.ffn\.2\.":             r"t2u_model.decoder.layers.\1.conv1d.conv2.",
                fr"^{t2u_decoder_key}\.layers\.([0-9]+)\.ffn\.layer_norm\.":         r"t2u_model.decoder.layers.\1.conv1d_layer_norm.",
                fr"^{t2u_decoder_key}\.layer_norm\.":                                r"t2u_model.decoder.layer_norm.",
                fr"^{t2u_decoder_key}\.output_projection\.":                         r"t2u_model.final_proj.",
                # fmt: on
            }
        )

    return key_map


load_unity_config = ConfigLoader[UnitYConfig](asset_store, unity_archs)


load_unity_model = ModelLoader[UnitYModel, UnitYConfig](
    asset_store,
    download_manager,
    load_unity_config,
    create_unity_model,
    convert_unity_checkpoint,
    restrict_checkpoints=False,
)


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
            card.field("num_units").as_(int),
            card.field("unit_langs").as_list(str),
            card.field("model_arch").as_(str),
        )


load_unity_unit_tokenizer = UnitYUnitTokenizerLoader(asset_store)


class GcmvnStatsLoader:
    """Loads GCMVN stats (mean & std) for ProsodyUnitY."""

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        """
        self.asset_store = asset_store

    def __call__(
        self, model_name_or_card: Union[str, AssetCard]
    ) -> Tuple[List[float], List[float]]:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        try:
            gcmvn_stats: Dict[str, List[float]] = card.field("gcmvn_stats").as_(dict)
        except AssetCardFieldNotFoundError:
            model_override = card.field("model_config").as_(dict)
            gcmvn_stats = model_override["gcmvn_stats"]

        return gcmvn_stats["mean"], gcmvn_stats["std"]


load_gcmvn_stats = GcmvnStatsLoader(asset_store)
