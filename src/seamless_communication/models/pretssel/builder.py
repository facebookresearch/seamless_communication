# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal, Optional

from fairseq2.assets import asset_store
from fairseq2.data import VocabularyInfo
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import StandardEmbedding, init_scaled_embedding
from fairseq2.nn.position_encoder import SinusoidalPositionEncoder
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    MultiheadAttention,
    StandardMultiheadAttention,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.typing import DataType, Device
from torch.nn import Conv1d

from seamless_communication.models.pretssel.ecapa_tdnn_builder import (
    EcapaTDNNBuilder,
    EcapaTDNNConfig,
    ecapa_tdnn_archs,
)
from seamless_communication.models.pretssel.pretssel_model import (
    PostNet,
    PretsselDecoderFrontend,
    PretsselEncoderFrontend,
    PretsselModel,
)
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.fft_decoder_layer import (
    Conv1dBlock,
    FeedForwardTransformerLayer,
)
from seamless_communication.models.unity.length_regulator import (
    VarianceAdaptor,
    VariancePredictor,
)
from seamless_communication.models.unity.t2u_builder import VariancePredictorConfig


@dataclass
class PretsselEncoderFrontendConfig:
    prosody_encoder_config: EcapaTDNNConfig
    dropout: float
    lang_embed_dim: Optional[int] = None


@dataclass
class FFTLayerConfig:
    attention_heads: int
    hidden_dim: int
    kernel_size: int
    dropout: float
    conv1d_dropout: float
    film_cond_dim: int
    use_film: bool = False


@dataclass
class PretsselDecoderFrontendConfig:
    upsampling_type: Literal["gaussian", "hard"]
    variance_predictor_config: VariancePredictorConfig
    add_variance_parallel: bool


@dataclass
class PostnetConfig:
    dropout: float
    layers: int
    conv_dim: int
    conv_kernel_size: int


@dataclass
class PretsselConfig:
    model_name_or_card: str
    encoder_frontend_config: PretsselEncoderFrontendConfig
    fft_layer_config: FFTLayerConfig
    decoder_frontend_config: PretsselDecoderFrontendConfig
    post_net_config: PostnetConfig
    vocab_info: VocabularyInfo
    model_dim: int
    max_seq_len: int
    encoder_layers: int
    decoder_layers: int
    output_dim: int


pretssel_archs = ArchitectureRegistry[PretsselConfig]("pretssel")

pretssel_arch = pretssel_archs.decorator


@pretssel_arch("base")
def _base_pretssel() -> PretsselConfig:
    prosody_encoder_config = ecapa_tdnn_archs.get_config("base")

    encoder_frontend_config = PretsselEncoderFrontendConfig(
        prosody_encoder_config=prosody_encoder_config,
        dropout=0.2,
        lang_embed_dim=64,
    )

    fft_layer_config = FFTLayerConfig(
        attention_heads=2,
        hidden_dim=1024,
        kernel_size=9,
        dropout=0.0,
        conv1d_dropout=0.2,
        use_film=True,
        film_cond_dim=576,
    )

    variance_predictor_config = VariancePredictorConfig(
        var_pred_hidden_dim=512,
        var_pred_kernel_size=5,
        var_pred_dropout=0.5,
        use_film=True,
        film_cond_dim=576,
    )

    decoder_frontend_config = PretsselDecoderFrontendConfig(
        upsampling_type="gaussian",
        variance_predictor_config=variance_predictor_config,
        add_variance_parallel=True,
    )

    post_net_config = PostnetConfig(
        dropout=0.5,
        layers=5,
        conv_dim=512,
        conv_kernel_size=5,
    )

    return PretsselConfig(
        "pretssel_v1",
        encoder_frontend_config,
        fft_layer_config,
        decoder_frontend_config,
        post_net_config,
        vocab_info=VocabularyInfo(
            size=10004, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        model_dim=256,
        max_seq_len=10000,
        encoder_layers=4,
        decoder_layers=4,
        output_dim=80,
    )


class PretsselBuilder:
    """
    Builder module for PRETSSEL model
    """

    config: PretsselConfig
    prosody_encoder_builder: EcapaTDNNBuilder
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: PretsselConfig,
        prosody_encoder_builder: EcapaTDNNBuilder,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param devicev:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.prosody_encoder_builder = prosody_encoder_builder

        self.device, self.dtype = device, dtype

    def build_embed_tokens(self) -> StandardEmbedding:
        """Build a unit embedding table."""

        return StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embedding_dim=self.config.model_dim,
            init_fn=init_scaled_embedding,
            device=self.device,
            dtype=self.dtype,
        )

    def build_fft(self, num_layers: int) -> FeedForwardTransformer:
        """Build a Transformer encoder."""

        layers = [self.build_fft_layer() for _ in range(num_layers)]

        return FeedForwardTransformer(
            layers,
            norm_order=TransformerNormOrder.POST,
            device=self.device,
            dtype=self.dtype,
        )

    def build_fft_layer(self) -> FeedForwardTransformerLayer:
        """Build a Transformer decoder layer."""

        self_attn = self.build_attention(self.config.fft_layer_config.attention_heads)

        conv1d = Conv1dBlock(
            self.config.model_dim,
            self.config.fft_layer_config.hidden_dim,
            self.config.fft_layer_config.kernel_size,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        return FeedForwardTransformerLayer(
            self_attn,
            conv1d,
            dropout_p=0.0,  # fairseq1 doesn't have this
            conv1d_dropout_p=self.config.fft_layer_config.conv1d_dropout,
            use_film=self.config.fft_layer_config.use_film,
            film_cond_dim=self.config.fft_layer_config.film_cond_dim,
            device=self.device,
            dtype=self.dtype,
        )

    def build_attention(self, num_heads: int) -> MultiheadAttention:
        """Build a Transformer multi-head attention layer."""

        sdpa = create_default_sdpa(attn_dropout_p=self.config.fft_layer_config.dropout)

        return StandardMultiheadAttention(
            self.config.model_dim,
            num_heads,
            sdpa=sdpa,
            device=self.device,
            dtype=self.dtype,
        )

    def build_variance_adaptor(
        self,
        decoder_frontend_config: PretsselDecoderFrontendConfig,
    ) -> VarianceAdaptor:
        """Build a variance adaptor module."""

        variance_predictor_config = decoder_frontend_config.variance_predictor_config

        pitch_predictor = VariancePredictor(
            self.config.model_dim,
            variance_predictor_config.var_pred_hidden_dim,
            variance_predictor_config.var_pred_kernel_size,
            variance_predictor_config.var_pred_dropout,
            use_film=variance_predictor_config.use_film,
            film_cond_dim=variance_predictor_config.film_cond_dim,
            device=self.device,
            dtype=self.dtype,
        )

        embed_pitch = Conv1d(
            1,
            self.config.model_dim,
            kernel_size=1,
            device=self.device,
            dtype=self.dtype,
        )

        vuv_predictor = VariancePredictor(
            self.config.model_dim,
            variance_predictor_config.var_pred_hidden_dim,
            variance_predictor_config.var_pred_kernel_size,
            variance_predictor_config.var_pred_dropout,
            use_film=variance_predictor_config.use_film,
            film_cond_dim=variance_predictor_config.film_cond_dim,
            device=self.device,
            dtype=self.dtype,
        )

        energy_predictor = VariancePredictor(
            self.config.model_dim,
            variance_predictor_config.var_pred_hidden_dim,
            variance_predictor_config.var_pred_kernel_size,
            variance_predictor_config.var_pred_dropout,
            use_film=variance_predictor_config.use_film,
            film_cond_dim=variance_predictor_config.film_cond_dim,
            device=self.device,
            dtype=self.dtype,
        )

        embed_energy = Conv1d(
            1,
            self.config.model_dim,
            kernel_size=1,
            device=self.device,
            dtype=self.dtype,
        )

        variance_adaptor = VarianceAdaptor(
            duration_predictor=None,
            pitch_predictor=pitch_predictor,
            embed_pitch=embed_pitch,
            vuv_predictor=vuv_predictor,
            energy_predictor=energy_predictor,
            embed_energy=embed_energy,
            add_variance_parallel=decoder_frontend_config.add_variance_parallel,
            upsampling_type=decoder_frontend_config.upsampling_type,
        )

        return variance_adaptor

    def build_model(self) -> PretsselModel:
        """Build a model."""
        prosody_encoder = self.prosody_encoder_builder.build_model()

        embed_tokens = self.build_embed_tokens()

        embed_positions = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
            _legacy_pad_idx=self.config.vocab_info.pad_idx,
            device=self.device,
        )

        model_card = asset_store.retrieve_card(self.config.model_name_or_card)
        langs = model_card.field("languages").as_list(str)
        lang_to_index = {l: i for i, l in enumerate(langs)}

        encoder_frontend = PretsselEncoderFrontend(
            prosody_encoder,
            embed_tokens,
            embed_positions,
            lang_to_index,
            lang_embed_dim=self.config.encoder_frontend_config.lang_embed_dim,
            dropout_p=self.config.encoder_frontend_config.dropout,
            device=self.device,
            dtype=self.dtype,
        )

        encoder = self.build_fft(self.config.encoder_layers)

        variance_adaptor = self.build_variance_adaptor(
            self.config.decoder_frontend_config
        )

        decoder_frontend = PretsselDecoderFrontend(
            variance_adaptor,
            embed_positions,
            device=self.device,
            dtype=self.dtype,
        )

        decoder = self.build_fft(self.config.decoder_layers)

        final_proj = Linear(
            self.config.model_dim,
            self.config.output_dim,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        postnet = PostNet(
            self.config.output_dim,
            self.config.post_net_config.conv_dim,
            self.config.post_net_config.conv_kernel_size,
            self.config.post_net_config.layers,
            self.config.post_net_config.dropout,
            device=self.device,
            dtype=self.dtype,
        )

        return PretsselModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            postnet,
            device=self.device,
            dtype=self.dtype,
        )


def create_pretssel_model(
    config: PretsselConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> PretsselModel:
    """Create a PretsselModel.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    prosody_encoder_builder = EcapaTDNNBuilder(
        config.encoder_frontend_config.prosody_encoder_config,
        device=device,
        dtype=dtype,
    )
    return PretsselBuilder(
        config, prosody_encoder_builder, device=device, dtype=dtype
    ).build_model()
