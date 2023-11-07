# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Literal, Optional, Union

from fairseq2.assets import download_manager
from fairseq2.assets.card import AssetCard
from fairseq2.data import VocabularyInfo
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import Embedding, StandardEmbedding, init_scaled_embedding
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
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.nllb.loader import NllbTokenizerLoader


from seamless_communication.assets import asset_store
from seamless_communication.models.unity.nar_decoder import NARTransformerDecoder
from seamless_communication.models.unity.nar_decoder_layer import (
    NARTransformerDecoderLayer,
    Conv1dBlock,
)
from seamless_communication.models.unity.nar_decoder_frontend import NARDecoderFrontend
from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer
from seamless_communication.models.unity.model import UnitYT2UModel, UnitYNART2UModel
from seamless_communication.models.unity.length_regulator import (
    VariancePredictor,
    VarianceAdaptor,
)


@dataclass
class VariancePredictorConfig:
    var_pred_hidden_dim: int
    var_pred_kernel_size: int
    var_pred_dropout: float


@dataclass
class NARDecoderFrontendConfig:
    subword_to_unit_upsampling_type: Literal["gaussian", "hard"]
    duration_predictor_config: VariancePredictorConfig
    pitch_predictor_config: Optional[VariancePredictorConfig]
    energy_predictor_config: Optional[VariancePredictorConfig]


@dataclass
class NARDecoderConfig:
    model_name_or_card: Union[str, AssetCard]
    char_vocabulary_size: int
    char_max_seq_len: int
    conv1d_kernel_size: int
    conv1d_inner_dim: int
    conv1d_dropout_p: float


@dataclass
class UnitYT2UConfig:
    """Holds the configuration of a UnitY T2U model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`"""

    model_dim: int
    """The dimensionality of the model."""

    unit_max_seq_len: int
    """The expected maximum unit sequence length."""

    target_vocab_info: VocabularyInfo
    """The target vocabulary information."""

    num_encoder_layers: int
    """The number of Transformer encoder layers."""

    num_decoder_layers: int
    """The number of Transformer decoder layers."""

    nar_decoder_frontend_config: Optional[NARDecoderFrontendConfig]
    """Non-autoregressive decoder front-end config."""

    nar_decoder_config: Optional[NARDecoderConfig]
    """Non-autoregressive decoder config."""

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
        target_vocab_info=VocabularyInfo(
            size=10082, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        num_encoder_layers=6,
        num_decoder_layers=6,
        nar_decoder_frontend_config=None,
        nar_decoder_config=None,
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
        target_vocab_info=VocabularyInfo(
            size=10082, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        num_encoder_layers=4,
        num_decoder_layers=4,
        nar_decoder_frontend_config=None,
        nar_decoder_config=None,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.1,
    )


@unity_t2u_arch("base_nar")
def _base_nar() -> UnitYT2UConfig:
    duration_predictor_config = VariancePredictorConfig(
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
    )

    nar_decoder_frontend_config = NARDecoderFrontendConfig(
        subword_to_unit_upsampling_type="hard",
        duration_predictor_config=duration_predictor_config,
        pitch_predictor_config=None,
        energy_predictor_config=None,
    )

    nar_decoder_config = NARDecoderConfig(
        model_name_or_card="seamlessM4T_v2_large",
        char_vocabulary_size=10943,
        char_max_seq_len=4096,
        conv1d_kernel_size=7,
        conv1d_inner_dim=1024,
        conv1d_dropout_p=0.1,
    )

    return UnitYT2UConfig(
        model_dim=1024,
        unit_max_seq_len=4096,
        target_vocab_info=VocabularyInfo(
            size=10082, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        num_encoder_layers=6,
        num_decoder_layers=6,
        nar_decoder_frontend_config=nar_decoder_frontend_config,
        nar_decoder_config=nar_decoder_config,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.0,
    )


class UnitYT2UBuilder:
    """Builds modules of an autoregressive UnitY T2U model.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: UnitYT2UConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYT2UConfig,
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

    def build_model(self) -> UnitYT2UModel:
        """Build an autoregressive UnitYT2U model."""

        embed_unit = self.build_unit_embedding()

        encoder = self.build_encoder()

        decoder = self.build_decoder()

        final_proj = TiedProjection(embed_unit.weight, bias=None)

        decoder_frontend = self.build_decoder_frontend(embed_unit)

        return UnitYT2UModel(
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.target_vocab_info,
        )

    def build_unit_embedding(self) -> StandardEmbedding:
        """Build a unit embedding table."""

        return StandardEmbedding(
            num_embeddings=self.config.target_vocab_info.size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.target_vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
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

    def build_decoder_frontend(self, embed_unit: Embedding) -> TransformerFrontend:
        """Build a Transformer decoder front-end."""

        pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.unit_max_seq_len,
            _legacy_pad_idx=self.config.target_vocab_info.pad_idx,
            device=self.device,
        )
        return TransformerEmbeddingFrontend(
            embed_unit,
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
            bias=True,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )


class UnitYNART2UBuilder:
    """Builds modules of an NAR UnitY T2U model.

    To tweak the architecture, you can derive from this class and override the
    corresponding methods.
    """

    config: UnitYT2UConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: UnitYT2UConfig,
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

    def build_model(self) -> UnitYNART2UModel:
        """Build a non-autoregressive UnitY T2U model."""

        embed_unit = self.build_unit_embedding()

        encoder = self.build_encoder()

        decoder = self.build_decoder()

        final_proj = TiedProjection(embed_unit.weight, bias=None)

        decoder_frontend = self.build_decoder_frontend(embed_unit)

        return UnitYNART2UModel(
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.target_vocab_info,
        )

    def build_unit_embedding(self) -> StandardEmbedding:
        """Build a unit embedding table."""

        return StandardEmbedding(
            num_embeddings=self.config.target_vocab_info.size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.target_vocab_info.pad_idx,
            init_fn=init_scaled_embedding,
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

    def build_variance_adaptor(
        self, nar_decoder_frontend_config: NARDecoderFrontendConfig
    ) -> VarianceAdaptor:
        """Build a variance adaptor module."""

        duration_predictor_config = (
            nar_decoder_frontend_config.duration_predictor_config
        )
        duration_predictor = VariancePredictor(
            self.config.model_dim,
            duration_predictor_config.var_pred_hidden_dim,
            duration_predictor_config.var_pred_kernel_size,
            duration_predictor_config.var_pred_dropout,
            device=self.device,
            dtype=self.dtype,
        )

        variance_adaptor = VarianceAdaptor(
            duration_predictor,
            pitch_predictor=None,
            energy_predictor=None,
        )

        return variance_adaptor

    def build_decoder_frontend(self, embed_unit: Embedding) -> NARDecoderFrontend:
        """Build a non-autoregressive decoder front-end."""

        assert self.config.nar_decoder_config is not None
        assert self.config.nar_decoder_frontend_config is not None

        unit_pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.unit_max_seq_len,
            _legacy_pad_idx=self.config.target_vocab_info.pad_idx,
            device=self.device,
        )

        char_tokenizer = load_unity_char_tokenizer(
            self.config.nar_decoder_config.model_name_or_card
        )

        variance_adaptor = self.build_variance_adaptor(
            self.config.nar_decoder_frontend_config
        )

        nllb_tokenizer = NllbTokenizerLoader(asset_store, download_manager)(
            self.config.nar_decoder_config.model_name_or_card
        )
        text_pad_idx = nllb_tokenizer.vocab_info.pad_idx

        char_pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.nar_decoder_config.char_max_seq_len,
            _legacy_pad_idx=text_pad_idx,
            device=self.device,
        )

        embed_char = StandardEmbedding(
            num_embeddings=self.config.nar_decoder_config.char_vocabulary_size,
            embedding_dim=self.config.model_dim,
            pad_idx=text_pad_idx,
            init_fn=init_scaled_embedding,
            device=self.device,
            dtype=self.dtype,
        )

        return NARDecoderFrontend(
            embed_unit,
            embed_char,
            nllb_tokenizer,
            char_tokenizer,
            unit_pos_encoder,
            char_pos_encoder,
            variance_adaptor,
            dropout_p=self.config.dropout_p,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder(self) -> NARTransformerDecoder:
        """Build a Transformer decoder."""

        num_layers = self.config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return NARTransformerDecoder(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> NARTransformerDecoderLayer:
        """Build a Transformer decoder layer."""

        assert self.config.nar_decoder_config is not None

        self_attn = self.build_attention(self.config.num_decoder_attn_heads)

        conv1d = Conv1dBlock(
            self.config.model_dim,
            self.config.nar_decoder_config.conv1d_inner_dim,
            self.config.nar_decoder_config.conv1d_kernel_size,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        return NARTransformerDecoderLayer(
            self_attn,
            conv1d,
            dropout_p=self.config.dropout_p,
            conv1d_dropout_p=self.config.nar_decoder_config.conv1d_dropout_p,
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
            bias=True,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )


def create_unity_t2u_model(
    config: UnitYT2UConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Union[UnitYT2UModel, UnitYNART2UModel]:
    """Create a UnitY T2U model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    if config.nar_decoder_config is None:
        return UnitYT2UBuilder(config, device=device, dtype=dtype).build_model()
    else:
        return UnitYNART2UBuilder(config, device=device, dtype=dtype).build_model()
