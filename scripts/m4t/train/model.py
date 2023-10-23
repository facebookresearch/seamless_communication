# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Dict, Any

import torch
from m4t_scripts.train.configs import CustomModelParams, ModelConfig

from seamless_communication.models.unity import (
    UnitYConfig,
    UnitYModel,
    load_unity_model,
    create_unity_model,
)
from seamless_communication.models.unity.loader import load_unity_config
from seamless_communication.models.unity import UnitYT2UConfig
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.models.nllb.builder import NllbConfig
from fairseq2.models.utils.checkpoint_loader import convert_model_state_dict
from fairseq2.models.wav2vec2.loader import Wav2Vec2Loader
from seamless_communication.models.unity.loader import UnitYLoader

from fairseq2.models.nllb.loader import NllbLoader

logger = logging.getLogger(__name__)


CPU_DEVICE = torch.device("cpu")


class ModelBuilder:
    def __init__(
        self,
        config: ModelConfig,
        dtype: torch.dtype = torch.float16,
        device: torch.device = CPU_DEVICE,
    ):
        self.config = config
        self.dtype = dtype
        self.device = device

    @classmethod
    def _sel_and_upd_prefix(cls, kv: Dict[str, Any], prefix: str, new_prefix: str = "") -> Dict[str, Any]:
        # fmt: off
        return {new_prefix + k[len(prefix):]: v for k, v in kv.items() if k.startswith(prefix)}
        # fmt: on

    @classmethod
    def _load_pretrained_w2v2_encoder(cls, model: UnitYModel, checkpoint_path: str) -> None:
        """Load w2v2 encoder model trained in fairseq1"""
        logger.info(f"Loading w2v2 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path)["model"]
        key_map = Wav2Vec2Loader._fairseq_key_map()
        key_map.update(
            {
                r"^encoder.layers\.([0-9]+)\.conv_module.batch_norm.": r"encoder.layers.\1.conv.batch_norm.",
                r"^encoder.layers\.([0-9]+)\.conv_module.depthwise_conv.": r"encoder.layers.\1.conv.depthwise_conv.",
                r"^encoder.layers\.([0-9]+)\.conv_module.pointwise_conv([0-9]+)\.": (
                    r"encoder.layers.\1.conv.pointwise_conv\2."
                ),
                r"^encoder.layers\.([0-9]+)\.conv_module.layer_norm.": r"encoder.layers.\1.conv_layer_norm.",
                r"^encoder.layers\.([0-9]+)\.ffn([0-9]+)\.layer_norm.": r"encoder.layers.\1.ffn\2_layer_norm.",
                r"^encoder.layers\.([0-9]+)\.ffn([0-9]+)\.w_1.": r"encoder.layers.\1.ffn\2.inner_proj.",
                r"^encoder.layers\.([0-9]+)\.ffn([0-9]+)\.w_2.": r"encoder.layers.\1.ffn\2.output_proj.",
                r"^encoder.layers\.([0-9]+)\.self_attn.linear_k\.": r"encoder.layers.\1.self_attn.k_proj.",
                r"^encoder.layers\.([0-9]+)\.self_attn.linear_q\.": r"encoder.layers.\1.self_attn.q_proj.",
                r"^encoder.layers\.([0-9]+)\.self_attn.linear_v\.": r"encoder.layers.\1.self_attn.v_proj.",
                r"^encoder.layers\.([0-9]+)\.self_attn.linear_out\.": r"encoder.layers.\1.self_attn.output_proj.",
                r"^encoder.layers\.([0-9]+)\.self_attn.linear_pos.weight": (
                    r"encoder.layers.\1.self_attn.sdpa.r_proj.weight"
                ),
                r"^encoder.layers\.([0-9]+)\.self_attn.pos_bias_u": r"encoder.layers.\1.self_attn.sdpa.u_bias",
                r"^encoder.layers\.([0-9]+)\.self_attn.pos_bias_v": r"encoder.layers.\1.self_attn.sdpa.v_bias",
                # overrides existing rule
                r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.": r"encoder.layers.\1.layer_norm.",
            }
        )
        state_dict = convert_model_state_dict(state_dict=state_dict, key_map=key_map)
        # w2v2_encoder in fairseq2 have encoder layer_norm set to None
        for rm_key in ["encoder.layer_norm.bias", "encoder.layer_norm.weight"]:
            del state_dict[rm_key]
        enc_state_dict = cls._sel_and_upd_prefix(kv=state_dict, prefix="encoder.")
        model.speech_encoder.inner.load_state_dict(enc_state_dict, strict=True)  # type: ignore
        logger.info(f"Loaded w2v2 encoder from {checkpoint_path}")

        enc_fronted_state_dict = cls._sel_and_upd_prefix(kv=state_dict, prefix="encoder_frontend.")  # noqa
        # TODO: reconcile discrepancies between fr1 and fr2 model designs
        #  fr1-based w2v2 checkpoints with conv positional encoders use relpos self attention
        #   this is not compatible with the fr2 model design
        # model.speech_encoder_frontend.load_state_dict(enc_fronted_state_dict)
        # logger.info(f"Loaded w2v2 encoder frontend from {checkpoint_path}")

    @classmethod
    def _load_pretrained_s2t_decoder(cls, model: UnitYModel, checkpoint_path: str) -> None:
        """Load NLLB decoder trained in fairseq1"""
        logger.info(f"Loading s2t decoder weights from {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path)["model"]
        except ModuleNotFoundError:
            logger.info("If seeing `No module named 'omegaconf'`, run `pip install omegaconf`")
            raise
        decoder_prefix = "decoder."
        shared_state_dict = cls._sel_and_upd_prefix(kv=state_dict, prefix="shared_decoder.", new_prefix=decoder_prefix)
        shared_state_dict = convert_model_state_dict(
            state_dict=shared_state_dict, key_map=NllbLoader._fairseq_key_map()
        )
        for rm_key in ["decoder.embed_positions._float_tensor", "decoder.version"]:
            del shared_state_dict[rm_key]
        decoder_state = cls._sel_and_upd_prefix(kv=shared_state_dict, prefix=decoder_prefix, new_prefix="")
        frontend_state = cls._sel_and_upd_prefix(kv=shared_state_dict, prefix="decoder_frontend.", new_prefix="")
        proj_state = cls._sel_and_upd_prefix(kv=shared_state_dict, prefix="final_proj.", new_prefix="")
        model.text_decoder_frontend.load_state_dict(frontend_state, strict=True)
        logger.info(f"Loaded s2t decoder frontend weights from {checkpoint_path}")
        model.text_decoder.load_state_dict(decoder_state, strict=True)
        logger.info(f"Loaded s2t decoder weights from {checkpoint_path}")
        model.final_proj.load_state_dict(proj_state, strict=True)
        logger.info(f"Loaded s2t decoder final_proj weights from {checkpoint_path}")

    @classmethod
    def _load_pretrained_t2u(cls, model: UnitYModel, model_config: UnitYConfig, checkpoint_path: str) -> None:
        logger.info(f"Loading t2u weights from {checkpoint_path}")
        t2u_model = model.t2u_model
        assert t2u_model is not None
        try:
            state_dict = torch.load(checkpoint_path)["model"]
        except ModuleNotFoundError:
            logger.info("If seeing `No module named 'omegaconf'`, run `pip install omegaconf`")
            raise
        state_dict = {k.replace("encoder.", "synthesizer_encoder."): v for k, v in state_dict.items()}
        state_dict = convert_model_state_dict(
            state_dict=state_dict, key_map=UnitYLoader._fairseq_key_map(config=model_config)
        )
        t2u_state_dict = cls._sel_and_upd_prefix(kv=state_dict, prefix="t2u_model.", new_prefix="")
        t2u_model.load_state_dict(t2u_state_dict)
        logger.info(f"Loaded t2u weights from {checkpoint_path}")

    def build_model(
        self,
    ) -> UnitYModel:
        config = self.config
        logger.info("Initializing model")
        if config.from_model is not None:
            logger.info(f"Loading model and weights from `{config.from_model}`")
            return load_unity_model(config.from_model, device=self.device, dtype=self.dtype)

        if config.from_model_config is not None:
            logger.info(f"Loading Unity config from `{config.from_model_config}`")
            model_config = load_unity_config(config.from_model_config)
        elif config.custom_params is not None:
            logger.info("Creating custom Unity config")
            model_config = self._build_custom_model_config()
        else:
            raise ValueError("One of params from_model, from_model_config or custom_params has to be set")
        logger.info("Building model")
        model = create_unity_model(config=model_config, dtype=self.dtype, device=self.device)

        if self.config.pretrained_w2v2_path is not None:
            self._load_pretrained_w2v2_encoder(model, self.config.pretrained_w2v2_path)

        if self.config.pretrained_s2t_decoder_path is not None:
            self._load_pretrained_s2t_decoder(model, self.config.pretrained_s2t_decoder_path)

        if self.config.pretrained_t2u_path is not None:
            self._load_pretrained_t2u(model, model_config, self.config.pretrained_t2u_path)

        return model

    def _build_custom_model_config(self) -> UnitYConfig:
        config = self.config.custom_params
        assert config is not None
        return UnitYConfig(
            model_dim=config.model_embed_dim,
            w2v2_encoder_config=Wav2Vec2EncoderConfig(
                model_dim=config.model_embed_dim,
                max_seq_len=4096,
                feature_dim=160,
                use_fbank=True,
                first_pass_dropout_p=0.0,
                layer_norm_features=config.w2v2_encoder_layers_layernorm_features,
                feature_extractor_layer_descs=[],
                feature_extractor_bias=False,
                feature_extractor_layer_norm_convs=False,
                feature_grad_scale=0,
                num_fbank_channels=80,
                fbank_stride=2,
                sample_fbank_every_k=1,
                pos_encoder_type=config.w2v2_pos_encoder_type,
                pos_encoder_depth=config.w2v2_pos_encoder_depth,
                pos_conv_kernel_size=config.w2v2_pos_conv_kernel_size,
                num_pos_conv_groups=config.w2v2_num_pos_conv_groups,
                use_conformer=config.w2v2_encoder_layers_use_conformer,
                num_encoder_layers=config.w2v2_encoder_layers,
                num_encoder_attn_heads=16,
                ffn_inner_dim=config.model_embed_dim * 4,
                dropout_p=0.0,
                attn_dropout_p=0.0,
                layer_drop_p=0.0,
                norm_order=TransformerNormOrder.POST,
                depthwise_conv_kernel_size=31,
            ),
            mt_model_config=NllbConfig(
                model_dim=config.model_embed_dim,
                max_seq_len=1024,
                vocabulary_size=config.nllb_vocabulary_size,  # num_tokens + langs + spec symbols
                pad_idx=0,
                num_encoder_layers=config.nllb_encoder_layers,
                num_decoder_layers=config.nllb_decoder_layers,
                num_encoder_attn_heads=16,
                num_decoder_attn_heads=16,
                ffn_inner_dim=config.model_embed_dim * 8,
                dropout_p=0.1,
            ),
            t2u_config=UnitYT2UConfig(
                model_dim=config.model_embed_dim,
                unit_max_seq_len=2048,
                unit_vocabulary_size=config.unit_vocabulary_size,
                unit_pad_idx=1,
                num_encoder_layers=config.t2u_encoder_layers,
                num_decoder_layers=config.t2u_decoder_layers,
                nar_decoder_frontend_config=None,
                nar_decoder_config=None,
                num_encoder_attn_heads=16,
                num_decoder_attn_heads=16,
                ffn_inner_dim=config.model_embed_dim * 8,
                dropout_p=0.1,
            ),
            use_text_encoder=True,
            use_conformer_adaptor=False,
            num_adaptor_layers=1,
            adaptor_kernel_size=8,
            adaptor_stride=8,
            adaptor_layer_norm=True,
            adaptor_dropout_p=0.1,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
    )
    config = ModelConfig(
        custom_params=CustomModelParams(
            nllb_vocabulary_size=256103,
        ),
        pretrained_w2v2_path="/fsx-ust/spopuri/datasets/PT_CKPT/w2v2/w2vbert2rpq_600m_al5.pt",
        pretrained_s2t_decoder_path="/fsx-ust/spopuri/datasets/PT_CKPT/S2T/S2T_M4T_V1_V1_cleaned.pt",
        pretrained_t2u_path="/fsx-ust/spopuri/datasets/PT_CKPT/T2U/V5_10K_p2_14_80K.pt",
    )
    builder = ModelBuilder(config=config)
    model = ModelBuilder(config=config).build_model()
