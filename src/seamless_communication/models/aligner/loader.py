# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, List, Mapping

import torch
from fairseq2.assets import get_asset_store, download_manager
# from fairseq2.models.utils import ConfigLoader, ModelLoader
from fairseq2.runtime.config_registry import ConfigRegistrar
from seamless_communication.models.aligner.builder import (
    UnitY2AlignmentConfig,
    # aligner_archs,
    create_unity2_alignment_model,
)
from seamless_communication.models.aligner.model import UnitY2AlignmentModel
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer


def convert_unity2_aligner_checkpoint(
    checkpoint: Mapping[str, Any], config: UnitY2AlignmentConfig
) -> Mapping[str, Any]:
    if (
        "model" in checkpoint
        and "alignment_encoder.t_conv.1.weight" in checkpoint["model"]
    ):
        return checkpoint

    alignment_frontend_statedict = {}
    text_emb_state_keymap = {"weight": "alignment_frontend.embed_text.weight"}
    for k, v in checkpoint["text_emb_state"].items():
        alignment_frontend_statedict[text_emb_state_keymap[k]] = v

    unit_emb_state_keymap = {"weight": "alignment_frontend.embed_unit.weight"}
    for k, v in checkpoint["unit_emb_state"].items():
        alignment_frontend_statedict[unit_emb_state_keymap[k]] = v

    alignment_encoder_state_dict = {}
    for k, v in checkpoint["aligner_state"].items():
        alignment_encoder_state_dict[f"alignment_encoder.{k}"] = v

    model_state = {
        **alignment_encoder_state_dict,
        **alignment_frontend_statedict,
    }

    char_embeds = model_state["alignment_frontend.embed_text.weight"]

    index_mapping = _get_char_index_mapping(config)
    vocab_size = len(index_mapping)
    char_embeds[torch.arange(vocab_size)] = char_embeds[index_mapping]

    checkpoint["model"] = model_state

    return checkpoint


def _get_char_index_mapping(config: UnitY2AlignmentConfig) -> List[int]:
    char_tokenizer = load_unity_char_tokenizer(config.model_name_or_card)
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


# load_unity2_alignment_config = ConfigLoader[UnitY2AlignmentConfig](
#     asset_store, aligner_archs
# )

# load_unity2_alignment_model = ModelLoader[UnitY2AlignmentModel, UnitY2AlignmentConfig](
#     asset_store,
#     download_manager,
#     load_unity2_alignment_config,
#     create_unity2_alignment_model,
#     convert_unity2_aligner_checkpoint,
#     restrict_checkpoints=False,
# )

def load_unity2_alignment_config(name: str) -> UnitY2AlignmentConfig:
    # ↔ ConfigLoader[UnitY2AlignmentConfig](asset_store, aligner_archs)
    return get_unity2_alignment_model_hub().get_model_config(name)

def load_unity2_alignment_model(
    name: str,
    *,
    device = None,
    dtype = None,
) -> UnitY2AlignmentModel:
    # ↔ ModelLoader[UnitY2AlignmentModel, UnitY2AlignmentConfig](...)
    return get_unity2_alignment_model_hub().load_model(name, device=device, dtype=dtype)

from fairseq2.assets import get_asset_store
from fairseq2.models.hub import ModelHub
from fairseq2.models.family import ModelFamily
from fairseq2.runtime.dependency import get_dependency_resolver


def get_unity2_alignment_model_hub() -> ModelHub[UnitY2AlignmentModel, UnitY2AlignmentConfig]:
    resolver = get_dependency_resolver()
    family = resolver.resolve(ModelFamily, key="unity2_alignment")
    return ModelHub(family, get_asset_store())