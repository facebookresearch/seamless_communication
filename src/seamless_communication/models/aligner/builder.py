# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union

import torch
from fairseq2.assets.card import AssetCard
from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import StandardEmbedding, init_scaled_embedding
from fairseq2.typing import DataType, Device

from seamless_communication.models.aligner.model import (
    UnitY2AlignmentEncoder,
    UnitY2AlignmentFrontend,
    UnitY2AlignmentModel,
)
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer
from seamless_communication.models.unity.loader import load_unity_unit_tokenizer


@dataclass
class AlignmentEncoderConfig:
    model_dim: int

    feat_dim: int

    num_text_layers: int

    num_feat_layers: int

    dropout: float

    temperature: float

    reduction_factor: int


@dataclass
class UnitY2AlignmentFrontendConfig:
    unit_vocab_info: VocabularyInfo

    text_vocab_size: int


@dataclass
class UnitY2AlignmentConfig:
    model_name_or_card: Union[str, AssetCard]

    alignment_encoder_config: AlignmentEncoderConfig

    alignment_frontend_config: UnitY2AlignmentFrontendConfig


aligner_archs = ArchitectureRegistry[UnitY2AlignmentConfig]("unity2_aligner")

aligner_arch = aligner_archs.decorator


@aligner_arch("nar_t2u_aligner")
def _aligner_nar_t2u() -> UnitY2AlignmentConfig:
    encoder_config = AlignmentEncoderConfig(
        model_dim=1024,
        feat_dim=1024,
        num_text_layers=2,
        num_feat_layers=3,
        dropout=0.1,
        temperature=1.0,
        reduction_factor=1,
    )

    frontend_config = UnitY2AlignmentFrontendConfig(
        unit_vocab_info=VocabularyInfo(
            size=10082, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        text_vocab_size=10943,
    )

    return UnitY2AlignmentConfig(
        model_name_or_card="nar_t2u_aligner",
        alignment_encoder_config=encoder_config,
        alignment_frontend_config=frontend_config,
    )


class UnitY2AlignmentBuilder:
    config: UnitY2AlignmentConfig
    device: Optional[Device]
    dtype: DataType

    def __init__(
        self,
        config: UnitY2AlignmentConfig,
        *,
        device: Optional[Device] = None,
        dtype: DataType = torch.float32,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.device, self.dtype = device, dtype

    def build_model(self) -> UnitY2AlignmentModel:
        alignment_frontend = self.build_alignment_frontend()

        alignment_encoder = self.build_alignment_encoder()

        return UnitY2AlignmentModel(alignment_frontend, alignment_encoder)

    def build_alignment_frontend(self) -> UnitY2AlignmentFrontend:
        text_tokenizer = load_unity_char_tokenizer(self.config.model_name_or_card)

        unit_tokenizer = load_unity_unit_tokenizer(self.config.model_name_or_card)

        embed_text = StandardEmbedding(
            num_embeddings=self.config.alignment_frontend_config.text_vocab_size,
            embedding_dim=self.config.alignment_encoder_config.model_dim,
            pad_idx=self.config.alignment_frontend_config.unit_vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
            device=self.device,
            dtype=self.dtype,
        )

        embed_unit = StandardEmbedding(
            num_embeddings=self.config.alignment_frontend_config.unit_vocab_info.size,
            embedding_dim=self.config.alignment_encoder_config.model_dim,
            pad_idx=self.config.alignment_frontend_config.unit_vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
            device=self.device,
            dtype=self.dtype,
        )

        return UnitY2AlignmentFrontend(
            embed_text, embed_unit, text_tokenizer, unit_tokenizer
        )

    def build_alignment_encoder(self, training: bool = False) -> UnitY2AlignmentEncoder:
        cfg = self.config.alignment_encoder_config
        alignment_encoder = UnitY2AlignmentEncoder(
            embed_dim=cfg.model_dim,
            feat_dim=cfg.feat_dim,
            text_layers=cfg.num_text_layers,
            feat_layers=cfg.num_feat_layers,
            dropout=cfg.dropout,
            temperature=cfg.temperature,
            reduction_factor=cfg.reduction_factor,
            dtype=self.dtype,
        )
        alignment_encoder.training = training

        return alignment_encoder


def create_unity2_alignment_model(
    config: UnitY2AlignmentConfig,
    device: Optional[Device] = None,
    dtype: DataType = torch.float32,
) -> UnitY2AlignmentModel:
    """Create a UnitY model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    unity2_aligner_builder = UnitY2AlignmentBuilder(
        config,
        device=device,
        dtype=dtype,
    )

    return unity2_aligner_builder.build_model()
