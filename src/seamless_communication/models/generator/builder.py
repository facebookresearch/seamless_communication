# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

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

from seamless_communication.models.generator.ecapa_tdnn_builder import (
    EcapaTDNNBuilder,
    EcapaTDNNConfig,
    ecapa_tdnn_archs,
)
from seamless_communication.models.generator.vocoder import (
    PretsselDecoderFrontend,
    PretsselEncoderFrontend,
    PretsselVocoder,
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
class VocoderConfig:
    """Holds the configuration of a Vocoder model."""

    encoder_frontend_config: PretsselEncoderFrontendConfig
    fft_layer_config: FFTLayerConfig
    decoder_frontend_config: PretsselDecoderFrontendConfig
    pn_conv_dim: int
    pn_layers: int
    pn_conv_kernel_size: int
    pn_dropout: float
    vocab_info: VocabularyInfo
    model_dim: int
    max_seq_len: int
    encoder_layers: int
    decoder_layers: int
    mel_dim: int
    langs: List  # type: ignore[type-arg]
    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    channels: int
    dimension: int
    n_filters: int
    ratios: List[int]
    norm: Literal["none", "weight_norm", "spectral_norm", "time_group_norm"]
    norm_params: Dict[str, Any]
    kernel_size: int
    last_kernel_size: int
    residual_kernel_size: int
    causal: bool
    pad_mode: str
    true_skip: bool
    compress: int
    lstm: int
    disable_norm_outer_blocks: int
    trim_right_ratio: float
    gcmvn_stats: Dict[str, List]  # type: ignore[type-arg]


vocoder_archs = ArchitectureRegistry[VocoderConfig]("vocoder_pretssel")


vocoder_arch = vocoder_archs.decorator


def pretssel_config() -> (
    Tuple[PretsselEncoderFrontendConfig, FFTLayerConfig, PretsselDecoderFrontendConfig]
):
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
    return (
        encoder_frontend_config,
        fft_layer_config,
        decoder_frontend_config,
    )


@vocoder_arch("16khz")
def _16khz_vocoder() -> VocoderConfig:
    (
        encoder_frontend_config,
        fft_layer_config,
        decoder_frontend_config,
    ) = pretssel_config()

    return VocoderConfig(
        encoder_frontend_config=encoder_frontend_config,
        fft_layer_config=fft_layer_config,
        decoder_frontend_config=decoder_frontend_config,
        pn_conv_dim=512,
        pn_layers=5,
        pn_conv_kernel_size=5,
        pn_dropout=0.5,
        vocab_info=VocabularyInfo(
            size=10004, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        model_dim=256,
        max_seq_len=10000,
        encoder_layers=4,
        decoder_layers=4,
        mel_dim=80,
        langs=[],
        upsample_rates=[5, 4, 4, 2],
        upsample_kernel_sizes=[10, 8, 8, 4],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        channels=1,
        dimension=128,
        n_filters=32,
        ratios=[8, 5, 4, 2],
        norm="weight_norm",
        norm_params={},
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        causal=False,
        pad_mode="constant",
        true_skip=True,
        compress=2,
        lstm=2,
        disable_norm_outer_blocks=0,
        trim_right_ratio=1.0,
        gcmvn_stats={},
    )


@vocoder_arch("24khz")
def _24khz_vocoder() -> VocoderConfig:
    (
        encoder_frontend_config,
        fft_layer_config,
        decoder_frontend_config,
    ) = pretssel_config()

    return VocoderConfig(
        encoder_frontend_config=encoder_frontend_config,
        fft_layer_config=fft_layer_config,
        decoder_frontend_config=decoder_frontend_config,
        pn_conv_dim=512,
        pn_layers=5,
        pn_conv_kernel_size=5,
        pn_dropout=0.5,
        vocab_info=VocabularyInfo(
            size=10004, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1
        ),
        model_dim=256,
        max_seq_len=10000,
        encoder_layers=4,
        decoder_layers=4,
        mel_dim=80,
        langs=[],
        upsample_rates=[5, 4, 4, 3],
        upsample_kernel_sizes=[10, 8, 8, 6],
        upsample_initial_channel=512,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        channels=1,
        dimension=128,
        n_filters=32,
        ratios=[8, 5, 4, 2],
        norm="weight_norm",
        norm_params={},
        kernel_size=7,
        last_kernel_size=7,
        residual_kernel_size=3,
        causal=False,
        pad_mode="constant",
        true_skip=True,
        compress=2,
        lstm=2,
        disable_norm_outer_blocks=0,
        trim_right_ratio=1.0,
        gcmvn_stats={},
    )


class PretsselVocoderBuilder:
    config: VocoderConfig
    prosody_encoder_builder: EcapaTDNNBuilder
    device: Optional[Device] = None
    dtype: Optional[DataType] = None

    def __init__(
        self,
        config: VocoderConfig,
        prosody_encoder_builder: EcapaTDNNBuilder,
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

        embed_pitch = Conv1d(1, self.config.model_dim, kernel_size=1)

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

        embed_energy = Conv1d(1, self.config.model_dim, kernel_size=1)

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

    def build_model(self) -> PretsselVocoder:
        """build the pretssel vocoder."""
        prosody_encoder = self.prosody_encoder_builder.build_model()
        embed_tokens = self.build_embed_tokens()

        embed_positions = SinusoidalPositionEncoder(
            self.config.model_dim,
            self.config.max_seq_len,
            _legacy_pad_idx=self.config.vocab_info.pad_idx,
            device=self.device,
        )
        lang_to_index = {l: i for i, l in enumerate(self.config.langs)}
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
            self.config.mel_dim,
            bias=True,
            device=self.device,
            dtype=self.dtype,
        )

        gcmvn_mean = gcmvn_std = None
        if self.config.gcmvn_stats is not None:
            gcmvn_mean = self.config.gcmvn_stats["mean"]
            gcmvn_std = self.config.gcmvn_stats["std"]

        vocoder = PretsselVocoder(
            encoder_frontend=encoder_frontend,
            encoder=encoder,
            decoder_frontend=decoder_frontend,
            decoder=decoder,
            final_proj=final_proj,
            pn_n_channels=self.config.pn_conv_dim,
            pn_kernel_size=self.config.pn_conv_kernel_size,
            pn_layers=self.config.pn_layers,
            pn_dropout=self.config.pn_dropout,
            upsample_rates=self.config.upsample_rates,
            upsample_kernel_sizes=self.config.upsample_kernel_sizes,
            upsample_initial_channel=self.config.upsample_initial_channel,
            resblock_kernel_sizes=self.config.resblock_kernel_sizes,
            resblock_dilation_sizes=self.config.resblock_dilation_sizes,
            channels=self.config.channels,
            dimension=self.config.dimension,
            n_filters=self.config.n_filters,
            ratios=self.config.ratios,
            norm=self.config.norm,
            norm_params=self.config.norm_params,
            kernel_size=self.config.kernel_size,
            last_kernel_size=self.config.last_kernel_size,
            residual_kernel_size=self.config.residual_kernel_size,
            causal=self.config.causal,
            pad_mode=self.config.pad_mode,
            true_skip=self.config.true_skip,
            compress=self.config.compress,
            lstm=self.config.lstm,
            disable_norm_outer_blocks=self.config.disable_norm_outer_blocks,
            trim_right_ratio=self.config.trim_right_ratio,
            gcmvn_mean=gcmvn_mean,
            gcmvn_std=gcmvn_std,
        )
        vocoder.to(dtype=self.dtype, device=self.device)
        return vocoder


def create_vocoder_model(
    config: VocoderConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> PretsselVocoder:
    prosody_encoder_builder = EcapaTDNNBuilder(
        config.encoder_frontend_config.prosody_encoder_config,
        device=device,
        dtype=dtype,
    )
    return PretsselVocoderBuilder(
        config, prosody_encoder_builder, device=device, dtype=dtype
    ).build_model()
