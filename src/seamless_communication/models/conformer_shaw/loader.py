# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Optional

import torch
from fairseq2.device import Device
from fairseq2.data_type import DataType
from fairseq2.assets import get_asset_store, download_manager
# from fairseq2.models.utils import ModelLoader 
from seamless_communication.checkpoint import convert_fairseq_checkpoint
from fairseq2.models.wav2vec2 import Wav2Vec2Config, get_wav2vec2_model_hub
from fairseq2.model_checkpoint import DelegatingModelCheckpointLoader

from fairseq2.models.wav2vec2 import get_wav2vec2_model_hub
from fairseq2.models.wav2vec2.model import Wav2Vec2Model

from seamless_communication.models.conformer_shaw.builder import (
    create_conformer_shaw_model,
)


def convert_conformer_shaw_checkpoint(
    checkpoint: Mapping[str, Any], config: Wav2Vec2Config
) -> Mapping[str, Any]:
    """Convert a fairseq conformer shaw checkpoint to fairseq2."""
    state_dict = checkpoint["model"]

    # Check if we have a fairseq2 checkpoint.
    if "final_target_proj.weight" in state_dict:
        return checkpoint

    for key in (
        "mlm_proj.weight",
        "mlm_proj.bias",
        "encoder.layer_norm.weight",
        "encoder.layer_norm.bias",
    ):
        if key in state_dict:
            del state_dict[key]

    state_dict["quantizer.num_updates"] = torch.zeros((), device="cpu")

    key_map = {
        # fmt: off
        r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":            r"encoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.rel_k_embedding\.":     r"encoder.layers.\1.self_attn.sdpa.rel_k_embed.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.depthwise_conv\.":    r"encoder.layers.\1.conv.depthwise_conv.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.layer_norm\.":        r"encoder.layers.\1.conv_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.layer_norm2\.":       r"encoder.layers.\1.conv.layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv1\.":   r"encoder.layers.\1.conv.pointwise_conv1.",
        r"^encoder\.layers\.([0-9]+)\.conv_module\.pointwise_conv2\.":   r"encoder.layers.\1.conv.pointwise_conv2.",
        r"^encoder\.layers\.([0-9]+)\.fc1\.":                            r"encoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc2\.":                            r"encoder.layers.\1.ffn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.layer_norm\.":           r"encoder.layers.\1.ffn\2_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_1\.":                  r"encoder.layers.\1.ffn\2.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.ffn(1|2)\.w_2\.":                  r"encoder.layers.\1.ffn\2.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":               r"encoder.layers.\1.layer_norm.",
        r"^encoder\.embed_tokens\.":                                     r"encoder_frontend.embed.",
        r"^encoder\.pos_conv\.0\.":                                      r"encoder_frontend.pos_encoder.conv.",
        r"^feature_extractor\.conv_layers\.([0-9]+)\.0\.":               r"encoder_frontend.feature_extractor.layers.\1.conv.",
        r"^feature_extractor\.conv_layers\.([0-9]+)\.2\.1\.":            r"encoder_frontend.feature_extractor.layers.\1.layer_norm.",
        r"^feature_extractor\.conv_layers\.0\.2\.":                      r"encoder_frontend.feature_extractor.layers.0.group_norm.",
        r"^layer_norm\.":                                                r"encoder_frontend.post_extract_layer_norm.",
        r"^post_extract_proj\.":                                         r"encoder_frontend.model_dim_proj.",
        r"^mask_emb":                                                    r"masker.temporal_mask_embed",
        r"^quantizer\.vars":                                             r"quantizer.entries",
        r"^quantizer\.weight_proj\.":                                    r"quantizer.entry_proj.",
        r"^project_q\.":                                                 r"final_target_proj.",
        # fmt: on
    }

    return convert_fairseq_checkpoint(checkpoint, key_map)


# load_conformer_shaw_model = ModelLoader[Wav2Vec2Model, Wav2Vec2Config](
#     asset_store,
#     download_manager,
#     load_wav2vec2_config,
#     create_conformer_shaw_model,
#     convert_conformer_shaw_checkpoint,
# )

def load_wav2vec2_config(name: str) -> Wav2Vec2Config:
    hub = get_wav2vec2_model_hub()
    return hub.get_model_config(name)

def load_conformer_shaw_model(
    name: str,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    # 1) Config via Hub (comme ConfigLoader)
    hub = get_wav2vec2_model_hub()
    cfg = hub.get_model_config(name)

    # 2) Création du modèle via ta fabrique dédiée (respecte create_conformer_shaw_model)
    model = create_conformer_shaw_model(cfg, device=device, dtype=dtype)

    # 3) Résoudre l’URI du checkpoint depuis la card (équivalent ModelLoader + asset_store)
    card = get_asset_store().retrieve_card(name)
    resources = card.field("resources").as_(list)
    ckpt_uri = None
    for res in resources:
        if isinstance(res, dict) and res.get("name") in ("checkpoint", "model", "weights"):
            ckpt_uri = res.get("uri")
            if ckpt_uri:
                break
    if ckpt_uri is None:
        raise ValueError(f"Card '{card.name}' has no checkpoint resource.")

    # 4) Charger + CONVERTIR l’état (respecte convert_conformer_shaw_checkpoint)
    loader = DelegatingModelCheckpointLoader()
    state = {
        k: t
        for k, t in loader.lazy_load(
            ckpt_uri,
            state_dict_converter=lambda s: convert_conformer_shaw_checkpoint(s, cfg),
        )
    }

    # 5) Injecter les poids (comme le faisait ModelLoader)
    model.load_state_dict(state, strict=False)
    return model