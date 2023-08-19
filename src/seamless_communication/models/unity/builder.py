# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional

from fairseq2.data import VocabularyInfo
from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.nllb import NllbBuilder, NllbConfig, nllb_archs
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from seamless_communication.models.unity.adaptor_block import (
    UnitYConformerAdaptorLayer,
    UnitYEncoderAdaptor,
    UnitYTransformerAdaptorLayer,
)
from seamless_communication.models.unity.model import UnitYModel, UnitYT2UModel
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderBuilder, Wav2Vec2EncoderConfig
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    StandardTransformerDecoder,
    StandardTransformerDecoderLayer,
    StandardTransformerEncoder,
    StandardTransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


@dataclass
class UnitYConfig:
    """Holds the configuration of a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`"""

    model_dim: int
    """The dimensionality of the model."""

    w2v2_encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the underlying wav2vec 2.0 encoder."""

    nllb_config: NllbConfig
    """The configuration of the underlying NLLB text encoder-decoder."""

    t2u_config: Optional["UnitYT2UConfig"]
    """The configuration of the UnitY T2U sub-model."""

    use_text_encoder: bool
    """If ``True``, uses an aligned NLLB encoder for the MT task."""

    use_conformer_adaptor: bool
    """If ``True``, uses a Conformer-based adaptor block."""

    num_adaptor_layers: int
    """The number of Transformer encoder layers in the adaptor block."""

    adaptor_kernel_size: int
    """The kernel size of 1D convolutions in the adaptor block."""

    adaptor_stride: int
    """The stride of 1D convolutions in the adaptor block."""

    adaptor_layer_norm: bool
    """If ``True``, applies Layer Normalization to outputs of the underlying
    encoder in the adaptor block."""

    adaptor_dropout_p: float
    """The dropout probability in Transformer layers of the adaptor block."""


unity_archs = ArchitectureRegistry[UnitYConfig]("unity")


unity_arch = unity_archs.marker


@unity_arch("base")
def _base() -> UnitYConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")

    nllb_config = nllb_archs.get_config("dense_1b")

    nllb_config.vocabulary_size = 256102  # NLLB-100

    t2u_config = unity_t2u_archs.get_config("base")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        nllb_config=nllb_config,
        t2u_config=t2u_config,
        use_text_encoder=True,
        use_conformer_adaptor=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


@unity_arch("medium")
def _medium() -> UnitYConfig:
    w2vbert_config = w2vbert_archs.get_config("300m")

    nllb_config = nllb_archs.get_config("dense_600m")

    nllb_config.vocabulary_size = 256206  # NLLB-200

    t2u_config = unity_t2u_archs.get_config("medium")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        nllb_config=nllb_config,
        t2u_config=t2u_config,
        use_text_encoder=True,
        use_conformer_adaptor=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


class UnitYBuilder:
    """Builds modules of a UnitY model.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: UnitYConfig
    w2v2_encoder_builder: Wav2Vec2EncoderBuilder
    nllb_builder: NllbBuilder
    t2u_builder: Optional["UnitYT2UBuilder"]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYConfig,
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder,
        nllb_builder: NllbBuilder,
        t2u_builder: Optional["UnitYT2UBuilder"],
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param w2v2_encoder_builder:
            The wav2vec 2.0 encoder builder.
        :param nllb_builder:
            The NLLB model builder.
        :param t2u_builder:
            The UnitY T2U model builder.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        if w2v2_encoder_builder.config.model_dim != config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `w2v2_encoder_builder.config` must be equal, but are {config.model_dim} and {w2v2_encoder_builder.config.model_dim} instead."
            )

        if nllb_builder.config.model_dim != config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `nllb_builder.config` must be equal, but are {config.model_dim} and {nllb_builder.config.model_dim} instead."
            )

        if t2u_builder is not None and t2u_builder.config.model_dim != config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `t2u_builder.config` must be equal, but are {config.model_dim} and {t2u_builder.config.model_dim} instead."
            )

        self.config = config
        self.w2v2_encoder_builder = w2v2_encoder_builder
        self.nllb_builder = nllb_builder
        self.t2u_builder = t2u_builder
        self.device = device
        self.dtype = dtype

    def build_model(self) -> UnitYModel:
        """Build a model."""
        text_embed = self.nllb_builder.build_embedding()

        speech_encoder_frontend = self.w2v2_encoder_builder.build_frontend()
        speech_encoder = self.build_speech_encoder()

        text_decoder_frontend = self.nllb_builder.build_frontend(text_embed)
        text_decoder = self.nllb_builder.build_decoder()

        if self.config.use_text_encoder:
            # We use shared embedding as in NLLB.
            text_encoder_frontend = text_decoder_frontend
            text_encoder = self.nllb_builder.build_encoder()
        else:
            text_encoder_frontend = None
            text_encoder = None

        final_proj = TiedProjection(text_embed.weight)

        if self.t2u_builder is None:
            t2u_model = None
        else:
            t2u_model = self.t2u_builder.build_model()

        return UnitYModel(
            speech_encoder_frontend,
            speech_encoder,
            text_encoder_frontend,
            text_encoder,
            text_decoder_frontend,
            text_decoder,
            final_proj,
            t2u_model,
            self.config.nllb_config.pad_idx,
        )

    def build_speech_encoder(self) -> TransformerEncoder:
        """Build a speech Transformer encoder."""
        w2v2_encoder = self.w2v2_encoder_builder.build_encoder()

        # For Conformer-based wav2vec 2.0 architectures (e.g. w2v-BERT), we
        # typically use a special type of adaptor layer.
        if not self.config.use_conformer_adaptor:
            build_adaptor_layer = self.build_adaptor_layer
        else:
            build_adaptor_layer = self.build_conformer_adaptor_layer

        num_layers = self.config.num_adaptor_layers

        layers = [build_adaptor_layer(i) for i in range(num_layers)]

        return UnitYEncoderAdaptor(
            w2v2_encoder,
            layers,
            self.config.adaptor_layer_norm,
            device=self.device,
            dtype=self.dtype,
        )

    def build_adaptor_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build a Transformer-based encoder adaptor layer."""
        self_attn = self.build_adaptor_attention(
            self.w2v2_encoder_builder.config.num_encoder_attn_heads
        )

        # Unlike wav2vec2, we use ReLU (i.e. standard FFN activation function)
        # instead of GELU.
        ffn = StandardFeedForwardNetwork(
            self.config.model_dim,
            self.w2v2_encoder_builder.config.ffn_inner_dim,
            device=self.device,
            dtype=self.dtype,
        )

        return UnitYTransformerAdaptorLayer(
            self_attn,
            ffn,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            self.config.adaptor_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_conformer_adaptor_layer(self, idx: int) -> TransformerEncoderLayer:
        """Build a Conformer-based encoder adaptor layer."""
        ffn1 = self.w2v2_encoder_builder.build_ffn(use_swish=True)

        # Empirically shown that, in adaptor layers, vanilla MHA performs better
        # than MHA with relative positional encoding.
        self_attn = self.build_adaptor_attention(
            self.w2v2_encoder_builder.config.num_encoder_attn_heads
        )

        conv = ConformerConvolution(
            self.w2v2_encoder_builder.config.model_dim,
            self.w2v2_encoder_builder.config.depthwise_conv_kernel_size,
            device=self.device,
            dtype=self.dtype,
        )

        ffn2 = self.w2v2_encoder_builder.build_ffn(use_swish=True)

        block = ConformerBlock(
            ffn1,
            self_attn,
            conv,
            ffn2,
            dropout_p=self.config.adaptor_dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

        layer_norm = idx == 0

        return UnitYConformerAdaptorLayer(
            block,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            layer_norm,
            device=self.device,
            dtype=self.dtype,
        )

    def build_adaptor_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer in adaptor block."""
        sdpa = create_default_sdpa(attn_dropout_p=self.config.adaptor_dropout_p)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )


def create_unity_model(
    config: UnitYConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> UnitYModel:
    """Create a UnitY model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    w2v2_encoder_builder = Wav2Vec2EncoderBuilder(
        config.w2v2_encoder_config, device, dtype
    )

    nllb_builder = NllbBuilder(config.nllb_config, device, dtype)

    if config.t2u_config is None:
        t2u_builder = None
    else:
        t2u_builder = UnitYT2UBuilder(config.t2u_config, device, dtype)

    unity_builder = UnitYBuilder(
        config, w2v2_encoder_builder, nllb_builder, t2u_builder, device, dtype
    )

    return unity_builder.build_model()


@dataclass
class UnitYT2UConfig:
    """Holds the configuration of a UnitY T2U model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`"""

    model_dim: int
    """The dimensionality of the model."""

    unit_max_seq_len: int
    """The expected maximum unit sequence length."""

    unit_vocabulary_size: int
    """The size of the unit vocabulary."""

    unit_pad_idx: Optional[int]
    """The index of the pad symbol in the unit vocabulary."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    num_encoder_attn_heads: int
    """The number of attention heads in Transformer encoder layers."""

    num_decoder_attn_heads: int
    """The number of attention heads in Transformer decoder layers."""

    ffn_inner_dim: int
    """The inner dimensionality of Transformer feed-forward networks."""

    dropout_p: float
    """The dropout probability in Transformer layers."""

    def update_unit_vocabulary(self, info: VocabularyInfo) -> None:
        """Update unit vocabulary configuration from ``info``."""
        self.unit_vocabulary_size, self.unit_pad_idx = info.size, info.pad_idx


unity_t2u_archs = ArchitectureRegistry[UnitYT2UConfig]("unity_t2u")


unity_t2u_arch = unity_t2u_archs.marker


@unity_t2u_arch("base")
def _base_t2u() -> UnitYT2UConfig:
    return UnitYT2UConfig(
        model_dim=1024,
        unit_max_seq_len=2048,
        unit_vocabulary_size=10082,
        unit_pad_idx=1,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    )


@unity_t2u_arch("medium")
def _medium_t2u() -> UnitYT2UConfig:
    return UnitYT2UConfig(
        model_dim=1024,
        unit_max_seq_len=2048,
        unit_vocabulary_size=10082,
        unit_pad_idx=1,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    )


class UnitYT2UBuilder:
    """Builds modules of a UnitY T2U model.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: UnitYT2UConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYT2UConfig,
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
        self.device = device
        self.dtype = dtype

    def build_model(self) -> UnitYT2UModel:
        """Build a model."""
        embed = self.build_embedding()

        encoder = self.build_encoder()

        decoder_frontend = self.build_decoder_frontend(embed)
        decoder = self.build_decoder()

        final_proj = TiedProjection(embed.weight)

        return UnitYT2UModel(
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.unit_pad_idx,
        )

    def build_embedding(self) -> Embedding:
        """Build a unit embedding table."""
        return Embedding(
            num_embeddings=self.config.unit_vocabulary_size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.unit_pad_idx,
            scaled=True,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder(self) -> Optional[TransformerEncoder]:
        """Build a Transformer encoder."""
        num_layers = self.config.num_encoder_layers
        if num_layers == 0:
            return None

        layers = [self.build_encoder_layer() for _ in range(num_layers)]

        return StandardTransformerEncoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_encoder_layer(self) -> TransformerEncoderLayer:
        """Build a Transformer encoder layer."""
        self_attn = self.build_attention(self.config.num_encoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerEncoderLayer(
            self_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_frontend(self, embed: Embedding) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""
        pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.unit_max_seq_len,
            _legacy_pad_idx=self.config.unit_pad_idx,
            device=self.device,
            dtype=self.dtype,
        )

        return TransformerEmbeddingFrontend(
            embed,
            pos_encoder,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> TransformerDecoder:
        """Build a Transformer decoder."""
        num_layers = self.config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return StandardTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> TransformerDecoderLayer:
        """Build a Transformer decoder layer."""
        self_attn = self.build_attention(self.config.num_decoder_attn_heads)

        encoder_decoder_attn = self.build_attention(self.config.num_decoder_attn_heads)

        ffn = self.build_ffn()

        return StandardTransformerDecoderLayer(
            self_attn,
            encoder_decoder_attn,
            ffn,
            dropout_p=self.config.dropout_p,
            norm_order=TransformerNormOrder.PRE,
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

    def build_ffn(self) -> FeedForwardNetwork:
        """Build a Transformer feed-forward network."""
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )


def create_unity_t2u_model(
    config: UnitYT2UConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> UnitYT2UModel:
    """Create a UnitY T2U model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    return UnitYT2UBuilder(config, device, dtype).build_model()
