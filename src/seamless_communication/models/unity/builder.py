# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Union

from fairseq2.data.vocabulary_info import VocabularyInfo
from fairseq2.models.conformer import ConformerBlock, ConformerConvolution
from fairseq2.models.nllb import NllbBuilder, NllbConfig, nllb_archs
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderBuilder, Wav2Vec2EncoderConfig
from fairseq2.nn.projection import TiedProjection
from fairseq2.nn.transformer import (
    FeedForwardNetwork,
    MultiheadAttention,
    StandardFeedForwardNetwork,
    StandardMultiheadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device, override
from torch.nn import GELU, ReLU

from seamless_communication.models.conformer_shaw import (
    ConformerShawEncoderBuilder,
    ConformerShawEncoderConfig,
    conformer_shaw_archs,
)
from seamless_communication.models.generator.ecapa_tdnn_builder import (
    EcapaTDNNBuilder,
    EcapaTDNNConfig,
    ecapa_tdnn_archs,
)
from seamless_communication.models.unity.adaptor_block import (
    UnitYConformerAdaptorLayer,
    UnitYEncoderAdaptor,
    UnitYTransformerAdaptorLayer,
)
from seamless_communication.models.unity.model import UnitYModel
from seamless_communication.models.unity.t2u_builder import (
    UnitYNART2UBuilder,
    UnitYT2UBuilder,
    UnitYT2UConfig,
    unity_t2u_archs,
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

    prosody_encoder_config: Optional[EcapaTDNNConfig]
    """The configuration of the expressive prosody encoder."""

    use_text_encoder: bool
    """If ``True``, uses an aligned MT encoder for the MT task."""

    use_text_decoder: bool
    """If ``False``, skips loading a text decoder, to be used with a Monotonic decoder."""

    use_conformer_adaptor: bool
    """If ``True``, uses a Conformer-based adaptor block."""

    use_gelu: bool
    """If ``True``, uses GELU activation function in feed-forward networks of
    adaptor blocks and decoder layers."""

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

unity_arch = unity_archs.decorator


@unity_arch("base")
def _base() -> UnitYConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")

    mt_model_config: NllbConfig = nllb_archs.get_config("dense_1b")

    mt_model_config.vocab_info.size = 256102  # NLLB-100

    t2u_config = unity_t2u_archs.get_config("base")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        mt_model_config=mt_model_config,
        t2u_config=t2u_config,
        prosody_encoder_config=None,
        use_text_encoder=True,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=False,
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

    mt_model_config.vocab_info.size = 256206  # NLLB-200

    t2u_config = unity_t2u_archs.get_config("medium")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=w2vbert_config.w2v2_config.encoder_config,
        mt_model_config=mt_model_config,
        t2u_config=t2u_config,
        prosody_encoder_config=None,
        use_text_encoder=True,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


@unity_arch("base_v2")
def _base_v2() -> UnitYConfig:
    conformer_shaw_encoder_config = conformer_shaw_archs.get_config("600m")

    mt_model_config: NllbConfig = nllb_archs.get_config("dense_1b")

    mt_model_config.vocab_info.size = 256102  # NLLB-100

    mt_model_config.max_seq_len = 4096

    t2u_config = unity_t2u_archs.get_config("base_nar")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=conformer_shaw_encoder_config,
        mt_model_config=mt_model_config,
        t2u_config=t2u_config,
        prosody_encoder_config=None,
        use_text_encoder=True,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


@unity_arch("expressivity_v2")
def _expressivity_v2() -> UnitYConfig:
    conformer_shaw_encoder_config = conformer_shaw_archs.get_config("600m")

    mt_model_config: NllbConfig = nllb_archs.get_config("dense_1b")

    mt_model_config.vocab_info.size = 256102  # NLLB-100

    mt_model_config.max_seq_len = 10000

    t2u_config = unity_t2u_archs.get_config("expressivity_nar")

    prosody_encoder_config = ecapa_tdnn_archs.get_config("base")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=conformer_shaw_encoder_config,
        mt_model_config=mt_model_config,
        t2u_config=t2u_config,
        prosody_encoder_config=prosody_encoder_config,
        use_text_encoder=False,
        use_text_decoder=True,
        use_conformer_adaptor=False,
        use_gelu=True,
        num_adaptor_layers=1,
        adaptor_kernel_size=8,
        adaptor_stride=8,
        adaptor_layer_norm=True,
        adaptor_dropout_p=0.1,
    )


def _build_seamless_nano_model_config(
    model_embed_dim: int,
    ffn_emb_dim_mult: int,
    feature_stride: int,
    text_decoder_layers: int,
    text_dict_size: int,
    unit_dict_size: int,
):
    num_fbank_channels = 80
    fbank_stride = feature_stride
    nllb_ffn_inner_dim = model_embed_dim * ffn_emb_dim_mult
    w2v2_ffn_inner_dim = model_embed_dim * 4
    w2v2_encoder_layers_layernorm_features: bool = False
    w2v2_pos_encoder_type = "relative"
    w2v2_pos_encoder_depth: int = 0
    w2v2_pos_conv_kernel_size: int = 0
    w2v2_num_pos_conv_groups: int = 0
    w2v2_encoder_layers: int = 6
    w2v2_encoder_layers_use_conformer: bool = True
    nllb_encoder_layers: int = 1
    nllb_decoder_layers: int = text_decoder_layers
    text_vocab_info = VocabularyInfo(
        size=text_dict_size,
        unk_idx=3,
        bos_idx=0,
        eos_idx=2,
        pad_idx=1,
    )
    unit_vocab_info = VocabularyInfo(
        size=unit_dict_size,
        unk_idx=0,
        bos_idx=0,
        eos_idx=0,
        pad_idx=0,  # not used
    )

    model_config = UnitYConfig(
        use_gelu=False,
        use_text_decoder=True,
        prosody_encoder_config=None,
        model_dim=model_embed_dim,
        w2v2_encoder_config=Wav2Vec2EncoderConfig(
            model_dim=model_embed_dim,
            max_seq_len=4096,
            feature_dim=num_fbank_channels * fbank_stride,
            use_fbank=True,
            first_pass_dropout_p=0.0,
            layer_norm_features=w2v2_encoder_layers_layernorm_features,
            feature_extractor_layer_descs=[],
            feature_extractor_bias=False,
            feature_extractor_layer_norm_convs=False,
            feature_grad_scale=0,
            num_fbank_channels=num_fbank_channels,
            fbank_stride=fbank_stride,
            sample_fbank_every_k=1,
            pos_encoder_type=w2v2_pos_encoder_type,
            pos_encoder_depth=w2v2_pos_encoder_depth,
            pos_conv_kernel_size=w2v2_pos_conv_kernel_size,
            num_pos_conv_groups=w2v2_num_pos_conv_groups,
            use_conformer=w2v2_encoder_layers_use_conformer,
            num_encoder_layers=w2v2_encoder_layers,
            num_encoder_attn_heads=16,
            ffn_inner_dim=w2v2_ffn_inner_dim,
            dropout_p=0.0,
            attn_dropout_p=0.0,
            layer_drop_p=0.0,
            norm_order=TransformerNormOrder.POST,
            depthwise_conv_kernel_size=31,
        ),
        mt_model_config=NllbConfig(
            model_dim=model_embed_dim,
            max_seq_len=1024,
            vocab_info=text_vocab_info,
            num_encoder_layers=nllb_encoder_layers,
            num_decoder_layers=nllb_decoder_layers,
            num_encoder_attn_heads=16,
            num_decoder_attn_heads=16,
            ffn_inner_dim=nllb_ffn_inner_dim,
            dropout_p=0.1,
        ),
        t2u_config=UnitYT2UConfig(
            use_gelu=False,
            char_pad_idx=0,
            use_prosody_proj=False,
            prosody_encoder_dim=0,
            nar_decoder_frontend_config=None,
            nar_decoder_config=None,
            model_dim=model_embed_dim,
            unit_max_seq_len=2048,
            target_vocab_info=unit_vocab_info,  # dummy
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_encoder_attn_heads=16,
            num_decoder_attn_heads=16,
            ffn_inner_dim=model_embed_dim * 8,
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
    return model_config


@unity_arch("seamless_micro")
def _seamless_micro() -> UnitYConfig:
    return _build_seamless_nano_model_config(
        model_embed_dim=512,
        ffn_emb_dim_mult=8,
        feature_stride=4,
        text_decoder_layers=3,
        text_dict_size=20010,
        unit_dict_size=10082,
    )


@unity_arch("seamless_nano")
def _seamless_nano() -> UnitYConfig:
    return _build_seamless_nano_model_config(
        model_embed_dim=256,
        ffn_emb_dim_mult=8,
        feature_stride=4,
        text_decoder_layers=3,
        text_dict_size=20010,
        unit_dict_size=10082,
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
    prosody_encoder_builder: Optional[EcapaTDNNBuilder]
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYConfig,
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder,
        mt_model_builder: NllbBuilder,
        t2u_builder: Union[UnitYT2UBuilder, UnitYNART2UBuilder, None],
        prosody_encoder_builder: Optional[EcapaTDNNBuilder],
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
                "`model_dim` and `model_dim` of `w2v2_encoder_builder.config` must be equal, "
                f"but are {config.model_dim} and {w2v2_encoder_builder.config.model_dim} instead."
            )

        if mt_model_builder.config.model_dim != config.model_dim:
            raise ValueError(
                "`model_dim` and `model_dim` of `mt_model_builder.config` must be equal, "
                f"but are {config.model_dim} and {mt_model_builder.config.model_dim} instead."
            )

        if t2u_builder is not None and t2u_builder.config.model_dim != config.model_dim:
            raise ValueError(
                "`model_dim` and `model_dim` of `t2u_builder.config` must be equal, "
                f"but are {config.model_dim} and {t2u_builder.config.model_dim} instead."
            )

        self.config = config

        self.w2v2_encoder_builder = w2v2_encoder_builder
        self.mt_model_builder = mt_model_builder
        self.t2u_builder = t2u_builder
        self.prosody_encoder_builder = prosody_encoder_builder

        self.device, self.dtype = device, dtype

    def build_model(self) -> UnitYModel:
        """Build a model."""
        speech_encoder_frontend = self.w2v2_encoder_builder.build_frontend()
        speech_encoder = self.build_speech_encoder()

        if self.config.use_text_encoder:
            text_embed = self.mt_model_builder.build_embedding()
            text_encoder_frontend = self.mt_model_builder.build_frontend(text_embed)
            text_encoder = self.mt_model_builder.build_encoder()
        else:
            text_embed = None
            text_encoder_frontend = None
            text_encoder = None

        if self.config.use_text_decoder:
            if text_embed is None:
                text_embed = self.mt_model_builder.build_embedding()

            if text_encoder_frontend is not None:
                # We use shared embedding as in NLLB.
                text_decoder_frontend = text_encoder_frontend
            else:
                text_decoder_frontend = self.mt_model_builder.build_frontend(text_embed)

            text_decoder = self.mt_model_builder.build_decoder()
            final_proj = TiedProjection(text_embed.weight, bias=None)
        else:
            text_decoder_frontend = None
            text_decoder = None
            final_proj = None

        if self.t2u_builder is None:
            t2u_model = None
        else:
            t2u_model = self.t2u_builder.build_model()

        if self.prosody_encoder_builder is None:
            prosody_encoder_model = None
        else:
            prosody_encoder_model = self.prosody_encoder_builder.build_model()

        return UnitYModel(
            speech_encoder_frontend,
            speech_encoder,
            text_encoder_frontend,
            text_encoder,
            text_decoder_frontend,
            text_decoder,
            final_proj,
            t2u_model,
            self.config.mt_model_config.vocab_info,
            prosody_encoder_model,
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

        ffn = StandardFeedForwardNetwork(
            self.config.model_dim,
            self.w2v2_encoder_builder.config.ffn_inner_dim,
            inner_activation=GELU() if self.config.use_gelu else ReLU(),
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


class NllbWithGELUBuilder(NllbBuilder):
    @override
    def build_ffn(self) -> FeedForwardNetwork:
        return StandardFeedForwardNetwork(
            self.config.model_dim,
            self.config.ffn_inner_dim,
            bias=True,
            inner_activation=GELU(),
            norm_order=TransformerNormOrder.PRE,
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
    if isinstance(config.w2v2_encoder_config, ConformerShawEncoderConfig):
        w2v2_encoder_builder: Wav2Vec2EncoderBuilder = ConformerShawEncoderBuilder(
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

    if config.prosody_encoder_config is None:
        prosody_encoder_builder = None
    else:
        prosody_encoder_builder = EcapaTDNNBuilder(
            config.prosody_encoder_config, device=device, dtype=dtype
        )

    if config.use_gelu:
        mt_model_builder: NllbBuilder = NllbWithGELUBuilder(
            config.mt_model_config, device=device, dtype=dtype
        )
    else:
        mt_model_builder = NllbBuilder(
            config.mt_model_config, device=device, dtype=dtype
        )

    unity_builder = UnitYBuilder(
        config,
        w2v2_encoder_builder,
        mt_model_builder,
        t2u_builder,
        prosody_encoder_builder,
        device=device,
        dtype=dtype,
    )

    return unity_builder.build_model()
