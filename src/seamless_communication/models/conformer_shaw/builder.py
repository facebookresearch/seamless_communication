# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Optional

from fairseq2.runtime.config_registry import ConfigRegistrar

from fairseq2.models.conformer import ConformerConvolution
# from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.w2vbert import get_w2vbert_model_hub
from fairseq2.models.wav2vec2 import (
    Wav2Vec2Factory,
    Wav2Vec2Config,
    Wav2Vec2EncoderFactory,
    Wav2Vec2EncoderConfig,
    # wav2vec2_arch,
)
from fairseq2.models.wav2vec2 import Wav2Vec2Model, Wav2Vec2EncoderFactory
from fairseq2.models.transformer import RelativePositionalEncoding, MultiheadAttention, RelativePositionSDPA, create_default_sdpa, IdentityBias, StandardMultiheadAttention, SDPA, ShawRelativePositionSDPA, create_default_sdpa
from fairseq2.nn.position_encoder import RotaryEncoder
from fairseq2.runtime.lazy import Lazy
from overrides import override as override
from fairseq2.device import Device
from fairseq2.data_type import DataType
from fairseq2.nn import init_bert_projection

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


# conformer_shaw_archs = ArchitectureRegistry[ConformerShawEncoderConfig](
#     "conformer_shaw"
# )

# conformer_shaw_arch = conformer_shaw_archs.decorator

def load_arch_conformer_shaw(container):
    arch = ConfigRegistrar(container, ConformerShawEncoderConfig)
    @arch("600m")
    def _conformer_shaw_600m_encoder() -> ConformerShawEncoderConfig:
        w2vbert_hub=get_w2vbert_model_hub()
        w2vbert_config = w2vbert_hub.get_arch_config("600m")
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


    @arch("conformer_shaw_600m")
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


class ConformerShawEncoderFactory(Wav2Vec2EncoderFactory):
    """
    Conformer + ShawRelativePositionSDPA + depthwise conv causale + layer_norm.
    """

    def __init__(self, config: ConformerShawEncoderConfig) -> None:
        super().__init__(config)
        self.config = config

        assert self._config.use_conformer, "This architecture only supports a Conformer."
        assert (
            self._config.pos_encoder_type == "shaw_relative"
        ), "This architecture only supports ShawRelativePositionSDPA."

        if self._config.shaw_rel_pos_sdpa_config is None:
            raise ValueError(
                "`shaw_rel_pos_sdpa_config` must be specified when `pos_encoder_type` is 'shaw_relative'."
            )

    # On garde le pipeline de la Factory par défaut (create_encoder, etc.),
    # mais on spécialise les deux points suivants: SDPA et la conv Conformer.

    def create_self_attention(
        self, lazy_rel_pos_encoding: Lazy[RelativePositionalEncoding]
    ) -> MultiheadAttention:
        cfg = self._config

        # Positional encoder (rotary ou rien) — inchangé par rapport à la factory
        if cfg.pos_encoder_type == "rotary":
            pos_encoder = RotaryEncoder(
                cfg.model_dim // cfg.num_encoder_attn_heads, cfg.max_seq_len
            )
        else:
            pos_encoder = None

        attn_bias = IdentityBias()

        # >>> ShawRelativePositionSDPA <<<
        # sdpa_base = create_default_sdpa(attn_bias, dropout_p=cfg.attn_dropout_p)

        s = cfg.shaw_rel_pos_sdpa_config
        sdpa: SDPA = ShawRelativePositionSDPA(
            cfg.model_dim,
            cfg.num_encoder_attn_heads,
            attn_bias,
            max_lhs_rel_pos=s.max_left_rel_pos,
            max_rhs_rel_pos=s.max_right_rel_pos,
            use_rel_pos_values=s.use_rel_pos_values,
            # inner_sdpa=sdpa_base,
        )

        return StandardMultiheadAttention(
            cfg.model_dim,
            cfg.num_encoder_attn_heads,
            sdpa,
            qkv_proj_init_fn=init_bert_projection,
            pos_encoder=pos_encoder,
            output_proj_init_fn=init_bert_projection,
        )

    def create_conformer_conv(self) -> ConformerConvolution:
        cfg = self._config
        # >>> profondeur causale + LayerNorm <<<
        return ConformerConvolution(
            cfg.model_dim,
            cfg.depthwise_conv_kernel_size,
            causal_depthwise_conv=True,
            norm_type="layer_norm",
        )

from fairseq2.models.wav2vec2 import Wav2Vec2Factory, Wav2Vec2Model, Wav2Vec2Config
from fairseq2.models.wav2vec2 import Wav2Vec2Frontend
from fairseq2.models.transformer import TransformerEncoder

class ConformerShawWav2Vec2Factory(Wav2Vec2Factory):
    def create_encoder_frontend(self) -> Wav2Vec2Frontend:
        cfg = self._config
        enc_factory = ConformerShawEncoderFactory(cfg.encoder_config)
        return enc_factory.create_encoder_frontend()

    def create_encoder(self) -> TransformerEncoder:
        cfg = self._config
        enc_factory = ConformerShawEncoderFactory(cfg.encoder_config)
        return enc_factory.create_encoder()
    
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
    # construit tout en CPU / dtype par défaut
    model = ConformerShawWav2Vec2Factory(config).create_model()

    # placement/typage unifié (optionnel)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)

    return model
