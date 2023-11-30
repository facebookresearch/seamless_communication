# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Literal, Optional, Union

from fairseq2.assets import asset_store, download_manager
from fairseq2.assets.card import AssetCard
from fairseq2.data import VocabularyInfo
from fairseq2.models.nllb.loader import NllbTokenizerLoader
from fairseq2.models.transformer import (
    TransformerEmbeddingFrontend,
    TransformerFrontend,
)
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import Embedding, StandardEmbedding, init_scaled_embedding
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2.nn.projection import Linear, Projection, TiedProjection
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
from torch.nn import GELU, ReLU

from seamless_communication.models.unity.char_tokenizer import load_unity_char_tokenizer
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.fft_decoder_layer import (
    Conv1dBlock,
    FeedForwardTransformerLayer,
)
from seamless_communication.models.unity.length_regulator import (
    VarianceAdaptor,
    VariancePredictor,
)
from seamless_communication.models.unity.model import UnitYNART2UModel, UnitYT2UModel
from seamless_communication.models.unity.nar_decoder_frontend import NARDecoderFrontend


@dataclass
class VariancePredictorConfig:
    var_pred_hidden_dim: int
    var_pred_kernel_size: int
    var_pred_dropout: float
    use_film: bool
    film_cond_dim: int


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
    use_film: bool
    film_cond_dim: int


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

    use_gelu: bool
    """If ``True``, uses GELU activation function in feed-forward networks."""

    char_pad_idx: int
    """The index of the pad symbol in the char vocabulary."""

    use_prosody_proj: bool
    """If ``True``, uses a prosody projection layer."""

    prosody_encoder_dim: int
    """The dimensionality of prosody encoder (e.g. ECAPA_TDNN) output"""


unity_t2u_archs = ArchitectureRegistry[UnitYT2UConfig]("unity_t2u")


unity_t2u_arch = unity_t2u_archs.decorator


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
        use_gelu=False,
        char_pad_idx=1,
        use_prosody_proj=False,
        prosody_encoder_dim=0,
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
        use_gelu=False,
        char_pad_idx=1,
        use_prosody_proj=False,
        prosody_encoder_dim=0,
    )


@unity_t2u_arch("base_nar")
def _base_nar() -> UnitYT2UConfig:
    duration_predictor_config = VariancePredictorConfig(
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        use_film=False,
        film_cond_dim=0,
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
        use_film=False,
        film_cond_dim=0,
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
        use_gelu=False,
        char_pad_idx=1,
        use_prosody_proj=False,
        prosody_encoder_dim=0,
    )


@unity_t2u_arch("expressivity_nar")
def _expressivity_nar() -> UnitYT2UConfig:
    duration_predictor_config = VariancePredictorConfig(
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
        use_film=True,
        film_cond_dim=512,
    )

    nar_decoder_frontend_config = NARDecoderFrontendConfig(
        subword_to_unit_upsampling_type="hard",
        duration_predictor_config=duration_predictor_config,
        pitch_predictor_config=None,
        energy_predictor_config=None,
    )

    nar_decoder_config = NARDecoderConfig(
        model_name_or_card="seamless_expressivity",
        char_vocabulary_size=10904,
        char_max_seq_len=10000,
        conv1d_kernel_size=7,
        conv1d_inner_dim=1024,
        conv1d_dropout_p=0.1,
        use_film=True,
        film_cond_dim=512,
    )

    return UnitYT2UConfig(
        model_dim=1024,
        unit_max_seq_len=10000,
        target_vocab_info=VocabularyInfo(
            size=10005, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        num_encoder_layers=4,
        num_decoder_layers=4,
        nar_decoder_frontend_config=nar_decoder_frontend_config,
        nar_decoder_config=nar_decoder_config,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=1024 * 8,
        dropout_p=0.0,
        use_gelu=True,
        char_pad_idx=1,
        use_prosody_proj=True,
        prosody_encoder_dim=512,
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

        prosody_proj = self.build_prosody_proj()

        return UnitYNART2UModel(
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.target_vocab_info,
            prosody_proj=prosody_proj,
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
            use_film=duration_predictor_config.use_film,
            film_cond_dim=duration_predictor_config.film_cond_dim,
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

        # The legacy pad idx should be the same as that of the unit_pos_encoder,
        # since in fairseq1 the pos encoder is shared between both char, units.
        char_pos_encoder = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.nar_decoder_config.char_max_seq_len,
            _legacy_pad_idx=self.config.target_vocab_info.pad_idx,
            device=self.device,
        )

        embed_char = StandardEmbedding(
            num_embeddings=self.config.nar_decoder_config.char_vocabulary_size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.char_pad_idx,
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

    def build_decoder(self) -> FeedForwardTransformer:
        """Build a Transformer decoder."""

        num_layers = self.config.num_decoder_layers

        layers = [self.build_decoder_layer() for _ in range(num_layers)]

        return FeedForwardTransformer(
            layers,
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_decoder_layer(self) -> FeedForwardTransformerLayer:
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

        return FeedForwardTransformerLayer(
            self_attn,
            conv1d,
            dropout_p=self.config.dropout_p,
            conv1d_dropout_p=self.config.nar_decoder_config.conv1d_dropout_p,
            use_film=self.config.nar_decoder_config.use_film,
            film_cond_dim=self.config.nar_decoder_config.film_cond_dim,
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
            inner_activation=GELU() if self.config.use_gelu else ReLU(),
            norm_order=TransformerNormOrder.PRE,
            device=self.device,
            dtype=self.dtype,
        )

    def build_prosody_proj(self) -> Optional[Projection]:
        """Build a prosody projection layer if needed"""

        if self.config.use_prosody_proj:
            return Linear(
                self.config.prosody_encoder_dim,
                self.config.model_dim,
                bias=True,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            return None


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
