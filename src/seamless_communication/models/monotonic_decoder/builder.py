# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import Embedding, StandardEmbedding, init_scaled_embedding
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device

from seamless_communication.models.monotonic_decoder.model import MonotonicDecoderModel
from seamless_communication.models.monotonic_decoder.monotonic_decoder import (
    MonotonicTransformerDecoder,
)
from seamless_communication.models.monotonic_decoder.monotonic_decoder_layer import (
    MonotonicTransformerDecoderLayer,
)
from seamless_communication.models.monotonic_decoder.p_choose import PChooseLayer


@dataclass
class MonotonicDecoderConfig:
    """Holds the configuration of an Monotonic Decoder model."""

    model_dim: int
    """The dimensionality of the model."""

    max_seq_len: int
    """The expected maximum sequence length."""

    vocab_info: VocabularyInfo
    """The vocabulary information."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    energy_bias_value: float
    """The value of the energy bias parameter to be added to the
    monotonic energy in the PChooseLayer."""

    monotonic_temperature: float
    """The parameter with which to divide the monotonic energy
    to compute p_choose."""

    num_monotonic_energy_layers: int
    """The number of layers in the EnergyProjection module."""

    pre_decision_ratio: int
    """The kernel size and stride of the average pooling
    in the PChooseLayer."""


monotonic_decoder_archs = ArchitectureRegistry[MonotonicDecoderConfig](
    "monotonic_decoder"
)

monotonic_decoder_arch = monotonic_decoder_archs.decorator


@monotonic_decoder_arch("dense_1b")
def _dense_1b() -> MonotonicDecoderConfig:
    return MonotonicDecoderConfig(
        model_dim=1024,
        max_seq_len=4096,
        vocab_info=VocabularyInfo(
            size=256102, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0
        ),
        num_decoder_layers=24,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
        energy_bias_value=-0.5,
        monotonic_temperature=0.2,
        num_monotonic_energy_layers=4,
        pre_decision_ratio=2,
    )


class MonotonicDecoderBuilder:
    """Builds modules of a Monotonic Decoder.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: MonotonicDecoderConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: MonotonicDecoderConfig,
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
        self.config = config

        self.device, self.dtype = device, dtype

    def build_model(self) -> MonotonicDecoderModel:
        text_embed = self.build_embedding()

        text_decoder_frontend = self.build_frontend(text_embed)

        text_decoder = self.build_decoder()

        final_proj = TiedProjection(text_embed.weight, bias=None)

        return MonotonicDecoderModel(
            text_decoder_frontend,
            text_decoder,
            final_proj,
        )

    def build_embedding(self) -> StandardEmbedding:
        """Build an embedding table."""
        return StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
            device=self.device,
            dtype=self.dtype,
        )

    def build_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
            _legacy_pad_idx=1,
            device=self.device,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> MonotonicTransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return MonotonicTransformerDecoder(
            layers,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> MonotonicTransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self.config.num_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(self.config.num_decoder_attn_heads)

        p_choose_layer = self.build_p_choose_layer(self.config.num_decoder_attn_heads)

        ffn = self.build_ffn()

        return MonotonicTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            p_choose_layer,
            ffn,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_p_choose_layer(self, num_heads: int) -> PChooseLayer:
        """Build a PChoose layer."""
        return PChooseLayer(
            self.config.model_dim,
            num_heads,
            self.config.energy_bias_value,
            self.config.monotonic_temperature,
            self.config.num_monotonic_energy_layers,
            self.config.pre_decision_ratio,
            device=self.device,
            dtype=self.dtype,
        )

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )


def create_monotonic_decoder_model(
    config: MonotonicDecoderConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> MonotonicDecoderModel:
    """Create an Monotonic Decoder model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return MonotonicDecoderBuilder(config, device=device, dtype=dtype).build_model()
