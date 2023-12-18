# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Optional

from fairseq2.models.conformer import ConformerConvolution
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import w2vbert_archs
from fairseq2.models.wav2vec2.builder import (
    Wav2Vec2Builder,
    Wav2Vec2Config,
    Wav2Vec2EncoderBuilder,
    Wav2Vec2EncoderConfig,
    wav2vec2_arch,
)
from fairseq2.models.wav2vec2.model import Wav2Vec2Model
from fairseq2.nn.transformer import SDPA, ShawRelativePositionSDPA, create_default_sdpa
from fairseq2.typing import DataType, Device


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
class ConformerShawEncoderConfig(Wav2Vec2EncoderConfig):
    """Holds the configuration of a conformer shaw encoder."""

    shaw_rel_pos_sdpa_config: Optional[ShawRelativePositionSDPAConfig]
    """The parameters for ShawRelativePositionSDPA."""


conformer_shaw_archs = ArchitectureRegistry[ConformerShawEncoderConfig](
    "conformer_shaw"
)

conformer_shaw_arch = conformer_shaw_archs.decorator


@conformer_shaw_arch("600m")
def _conformer_shaw_600m_encoder() -> ConformerShawEncoderConfig:
    w2vbert_config = w2vbert_archs.get_config("600m")
    w2v2_encoder_config = w2vbert_config.w2v2_config.encoder_config
    sdpa_config = ShawRelativePositionSDPAConfig(
        max_left_rel_pos=64,
        max_right_rel_pos=8,
        use_rel_pos_values=False,
    )
    conformer_shaw_encoder_config = ConformerShawEncoderConfig(
        **asdict(w2v2_encoder_config),
        shaw_rel_pos_sdpa_config=sdpa_config,
    )
    conformer_shaw_encoder_config.pos_encoder_type = "shaw_relative"
    return conformer_shaw_encoder_config


@wav2vec2_arch("conformer_shaw_600m")
def _conformer_shaw_600m() -> Wav2Vec2Config:
    encoder_config = _conformer_shaw_600m_encoder()

    return Wav2Vec2Config(
        encoder_config,
        final_dim=768,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=768,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2.0, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.2,
    )


class ConformerShawEncoderBuilder(Wav2Vec2EncoderBuilder):
    """
    Builds modules of a `ConformerShawEncoderBuilder`.

    This is a Conformer architecture with these differences:
    - ShawRelativePositionSDPA as the SDPA.
    - ConformerConvolution with causal depthwise convolution
    and norm_type "layer_norm".
    """

    config: ConformerShawEncoderConfig

    def __init__(
        self,
        config: ConformerShawEncoderConfig,
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

        assert self.config.use_conformer, "This architecture only supports a Conformer."
        assert (
            self.config.pos_encoder_type == "shaw_relative"
        ), "This architecture only supports ShawRelativePositionSDPA."

    def build_sdpa(self) -> SDPA:
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

    def build_conformer_conv(self) -> ConformerConvolution:
        return ConformerConvolution(
            self.config.model_dim,
            self.config.depthwise_conv_kernel_size,
            causal_depthwise_conv=True,
            norm_type="layer_norm",
            device=self.device,
            dtype=self.dtype,
        )


def create_conformer_shaw_model(
    config: Wav2Vec2Config,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    """Create a conformer shaw model.

    :param config:
        The configuration.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    assert isinstance(config.encoder_config, ConformerShawEncoderConfig)

    encoder_builder = ConformerShawEncoderBuilder(
        config.encoder_config, device=device, dtype=dtype
    )

    builder = Wav2Vec2Builder(config, encoder_builder, device=device, dtype=dtype)

    return builder.build_model()
