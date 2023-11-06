# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Union, Optional

from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.nllb import NllbBuilder, NllbConfig, nllb_archs
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderBuilder, Wav2Vec2EncoderConfig
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer import (
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device


from seamless_communication.models.unity.adaptor_block import (
    UnitYConformerAdaptorLayer,
    UnitYEncoderAdaptor,
    UnitYTransformerAdaptorLayer,
)
from seamless_communication.models.unity.model import UnitYModel
from seamless_communication.models.unity.t2u_builder import (
    UnitYT2UBuilder,
    UnitYNART2UBuilder,
    UnitYT2UConfig,
    unity_t2u_archs,
)
from seamless_communication.models.wav2vec2_chunk import (
    wav2vec2_chunk_archs,
    Wav2Vec2ChunkEncoderBuilder,
    Wav2Vec2ChunkEncoderConfig,
)


@dataclass
class UnitYConfig:
    """Holds the configuration of a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`"""

    model_dim: int
    """The dimensionality of the model."""

    w2v2_encoder_config: Wav2Vec2EncoderConfig
    """The configuration of the underlying wav2vec 2.0 encoder."""

    mt_model_config: NllbConfig
    """The configuration of the underlying MT text encoder-decoder."""

    t2u_config: Optional[UnitYT2UConfig]
    """The configuration of the UnitY T2U sub-model."""

    use_text_encoder: bool
    """If ``True``, uses an aligned MT encoder for the MT task."""

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

    mt_model_config: NllbConfig = nllb_archs.get_config("dense_1b")

    mt_model_config.vocabulary_size = 256102  # NLLB-100

    t2u_config = unity_t2u_archs.get_config("base")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        mt_model_config=mt_model_config,
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

    mt_model_config: NllbConfig = nllb_archs.get_config("dense_600m")

    mt_model_config.vocabulary_size = 256206  # NLLB-200

    t2u_config = unity_t2u_archs.get_config("medium")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        mt_model_config=mt_model_config,
        t2u_config=t2u_config,
        use_text_encoder=True,
        use_conformer_adaptor=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


@unity_arch("base_v2")
def _base_v2() -> UnitYConfig:
    w2v2_chunk_encoder_config = wav2vec2_chunk_archs.get_config("600m")

    mt_model_config: NllbConfig = nllb_archs.get_config("dense_1b")

    mt_model_config.vocabulary_size = 256102  # NLLB-100

    mt_model_config.max_seq_len = 4096

    t2u_config = unity_t2u_archs.get_config("base_nar")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2v2_chunk_encoder_config,
        mt_model_config=mt_model_config,
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
    mt_model_builder: NllbBuilder
    t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYConfig,
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder,
        mt_model_builder: NllbBuilder,
        t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param w2v2_encoder_builder:
            The wav2vec 2.0 encoder builder.
        :param mt_model_builder:
            The MT model builder.
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

        if mt_model_builder.config.model_dim != config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `mt_model_builder.config` must be equal, but are {config.model_dim} and {mt_model_builder.config.model_dim} instead."
            )

        if t2u_builder is not None and t2u_builder.config.model_dim != config.model_dim:
            raise ValueError(
                f"`model_dim` and `model_dim` of `t2u_builder.config` must be equal, but are {config.model_dim} and {t2u_builder.config.model_dim} instead."
            )

        self.config = config

        self.w2v2_encoder_builder = w2v2_encoder_builder
        self.mt_model_builder = mt_model_builder
        self.t2u_builder = t2u_builder

        self.device, self.dtype = device, dtype

    def build_model(self) -> UnitYModel:
        """Build a model."""
        text_embed = self.mt_model_builder.build_embedding()

        speech_encoder_frontend = self.w2v2_encoder_builder.build_frontend()
        speech_encoder = self.build_speech_encoder()

        text_decoder_frontend = self.mt_model_builder.build_frontend(text_embed)
        text_decoder = self.mt_model_builder.build_decoder()

        if self.config.use_text_encoder:
            # We use shared embedding as in NLLB.
            text_encoder_frontend = text_decoder_frontend
            text_encoder = self.mt_model_builder.build_encoder()
        else:
            text_encoder_frontend = None
            text_encoder = None

        final_proj = TiedProjection(text_embed.weight, bias=None)

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
            self.config.mt_model_config.pad_idx,
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
            inner_layer_norm=self.config.adaptor_layer_norm,
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
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        return UnitYTransformerAdaptorLayer(
            self_attn,
            ffn,
            self.config.adaptor_kernel_size,
            self.config.adaptor_stride,
            dropout_p=self.config.adaptor_dropout_p,
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
            layer_norm=layer_norm,
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
    if isinstance(config.w2v2_encoder_config, Wav2Vec2ChunkEncoderConfig):
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder = Wav2Vec2ChunkEncoderBuilder(
            config.w2v2_encoder_config, device=device, dtype=dtype
        )
    else:
        w2v2_encoder_builder = Wav2Vec2EncoderBuilder(
            config.w2v2_encoder_config, device=device, dtype=dtype
        )

    t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None]

    if config.t2u_config is None:
        t2u_builder = None
    elif config.t2u_config.nar_decoder_config is None:
        t2u_builder = UnitYT2UBuilder(config.t2u_config, device=device, dtype=dtype)
    else:
        t2u_builder = UnitYNART2UBuilder(config.t2u_config, device=device, dtype=dtype)

    mt_model_builder = NllbBuilder(config.mt_model_config, device=device, dtype=dtype)
    unity_builder = UnitYBuilder(
        config,
        w2v2_encoder_builder,
        mt_model_builder,
        t2u_builder,
        device=device,
        dtype=dtype,
    )

    return unity_builder.build_model()
