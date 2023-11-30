# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Literal, Optional

from fairseq2.models.conformer import ConformerConvolution
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2.builder import (
    Wav2Vec2EncoderBuilder,
    Wav2Vec2EncoderConfig,
)
from fairseq2.nn.transformer import SDPA, ShawRelativePositionSDPA, create_default_sdpa
from fairseq2.typing import DataType, Device

from seamless_communication.models.wav2vec2_chunk.encoder import ChunkTransformerEncoder


@dataclass
class ShawRelativePositionSDPAConfig:
    """Holds the configuration of the :class:ShawRelativePositionSDPA module."""

    max_left_rel_pos: int
    """The left clipping value for relative positions."""

    max_right_rel_pos: Optional[int]
    """The right clipping value for relative positions."""

    use_rel_pos_values: bool = False
    """If True, also uses relative position values to compute relative attention."""


@dataclass
class Wav2Vec2ChunkEncoderConfig(Wav2Vec2EncoderConfig):
    """Holds the configuration of a wav2vec2 chunk encoder."""

    causal_depthwise_conv: bool
    """If True, uses a causal depthwise convolution similar to that described in
    Section 2.1 of :cite:t:`https://doi.org/10.48550/arxiv.1609.03499`."""

    conv_norm_type: Literal["batch_norm", "layer_norm"]
    """The type of normalization to use in the Conformer convolution module."""

    shaw_rel_pos_sdpa_config: Optional[ShawRelativePositionSDPAConfig]
    """The parameters for ShawRelativePositionSDPA."""

    chunk_size: int
    """The size of each chunk."""

    left_chunk_num: int
    """Number of chunks on the left up to which lookahead is allowed."""

    right_chunk_num: int
    """Number of chunks on the right up to which lookahead is allowed."""


wav2vec2_chunk_archs = ArchitectureRegistry[Wav2Vec2ChunkEncoderConfig](
    "wav2vec2_chunk"
)

wav2vec2_chunk_arch = wav2vec2_chunk_archs.decorator


@wav2vec2_chunk_arch("600m")
def _encoder_600m() -> Wav2Vec2ChunkEncoderConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")
    w2v2_encoder_config = w2vbert_config.w2v2_config.encoder_config
    sdpa_config = ShawRelativePositionSDPAConfig(
        max_left_rel_pos=64,
        max_right_rel_pos=8,
        use_rel_pos_values=False,
    )
    w2v2_chunk_encoder_config = Wav2Vec2ChunkEncoderConfig(
        **asdict(w2v2_encoder_config),
        causal_depthwise_conv=True,
        conv_norm_type="layer_norm",
        shaw_rel_pos_sdpa_config=sdpa_config,
        chunk_size=10000,
        left_chunk_num=128,
        right_chunk_num=0,
    )
    w2v2_chunk_encoder_config.pos_encoder_type = "shaw_relative"
    return w2v2_chunk_encoder_config


class Wav2Vec2ChunkEncoderBuilder(Wav2Vec2EncoderBuilder):
    config: Wav2Vec2ChunkEncoderConfig

    def __init__(
        self,
        config: Wav2Vec2ChunkEncoderConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        super().__init__(config, device=device, dtype=dtype)

        assert (
            self.config.use_conformer
        ), "Currently we only support the ChunkConformerBlock."

    def build_encoder(self) -> ChunkTransformerEncoder:
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return ChunkTransformerEncoder(
            layers,
            self.config.chunk_size,
            self.config.left_chunk_num,
            self.config.right_chunk_num,
            dropout_p=self.config.dropout_p,
            layer_drop_p=self.config.layer_drop_p,
        )

    def build_sdpa(self) -> SDPA:
        if self.config.pos_encoder_type == "shaw_relative":
            if self.config.shaw_rel_pos_sdpa_config is None:
                raise ValueError(
                    "`shaw_rel_pos_sdpa_config` must be specified when `pos_encoder_type` is 'shaw_relative'."
                )

            sdpa = create_default_sdpa(attn_dropout_p=self.config.attn_dropout_p)

            sdpa_config = self.config.shaw_rel_pos_sdpa_config

            return ShawRelativePositionSDPA(
                self.config.model_dim,
                self.config.num_encoder_attn_heads,
                sdpa_config.max_left_rel_pos,
                max_right_rel_pos=sdpa_config.max_right_rel_pos,
                use_rel_pos_values=sdpa_config.use_rel_pos_values,
                inner_sdpa=sdpa,
                device=self.device,
                dtype=self.dtype,
            )

        return super().build_sdpa()

    def build_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            causal_depthwise_conv=self.config.causal_depthwise_conv,
            norm_type=self.config.conv_norm_type,
            device=self.device,
            dtype=self.dtype,
        )
