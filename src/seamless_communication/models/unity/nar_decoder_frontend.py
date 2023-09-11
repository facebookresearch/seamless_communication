# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, final

from torch import Tensor
from torch.nn import Dropout, Module, Parameter

from fairseq2.data.text import TextTokenizer
from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm, StandardLayerNorm
from fairseq2.typing import DataType, Device, finaloverride
from fairseq2.nn.position_encoder import PositionEncoder

from seamless_communication.models.vocoder import VariancePredictor
from seamless_communication.models.unity.aligner import AlignmentEncoder
from seamless_communication.models.unity.length_regulator import (
    GaussianUpsampling,
    HardUpsampling,
)

import torch


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
    layer_norm: bool


class VarianceAdaptor(Module):
    def __init__(
        self,
        model_dim: int,
        upsampling_type: str,
        duration_predictor_config: VariancePredictorConfig,
        pitch_predictor_config: Optional[VariancePredictorConfig] = None,
        energy_predictor_config: Optional[VariancePredictorConfig] = None,
    ):
        super().__init__()
        if upsampling_type == "gaussian":
            self.gaussian_upsampling = GaussianUpsampling()
        else:
            self.register_module("gaussian_upsampling", None)

        if upsampling_type == "hard":
            self.hard_upsampling = HardUpsampling()
        else:
            self.register_module("hard_upsampling", None)

        self.duration_predictor = VariancePredictor(
            model_dim,
            duration_predictor_config.var_pred_hidden_dim,
            duration_predictor_config.var_pred_kernel_size,
            duration_predictor_config.var_pred_dropout,
        )

        if pitch_predictor_config:
            self.pitch_predictor = VariancePredictor(
                model_dim,
                pitch_predictor_config.var_pred_hidden_dim,
                pitch_predictor_config.var_pred_kernel_size,
                pitch_predictor_config.var_pred_dropout,
            )
        else:
            self.register_module("pitch_predictor", None)

        if energy_predictor_config:
            self.energy_predictor = VariancePredictor(
                model_dim,
                energy_predictor_config.var_pred_hidden_dim,
                energy_predictor_config.var_pred_kernel_size,
                energy_predictor_config.var_pred_dropout,
            )
        else:
            self.register_module("energy_predictor", None)

    def forward(self):
        pass


@final
class NARDecoderFrontend(TransformerFrontend):
    """Represents a NAR Decoder front-end."""

    def __init__(
        self,
        embed: Embedding,
        embed_char: Embedding,
        text_tokenizer: TextTokenizer,
        char_tokenizer: TextTokenizer,
        pos_encoder: Optional[PositionEncoder],
        decoder_frontend_config: NARDecoderFrontendConfig,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        model_dim = embed.embedding_dim

        super().__init__(model_dim)

        self.embed = embed
        self.embed_char = embed_char
        self.text_tokenizer = text_tokenizer
        self.char_tokenizer = char_tokenizer

        # training: alignment encoder
        if self.training:
            self.alignment_encoder = AlignmentEncoder()
        else:
            self.register_module("alignment_encoder", None)

        if pos_encoder is not None:
            if pos_encoder.encoding_dim != model_dim:
                raise ValueError(
                    f"`encoding_dim` of `pos_encoder` and `embedding_dim` of `embed` must be equal, but are {pos_encoder.encoding_dim} and {model_dim} instead."
                )

            self.pos_encoder = pos_encoder
        else:
            self.register_module("pos_encoder", None)

        self.pos_emb_alpha = Parameter(torch.ones(1))
        self.pos_emb_alpha_char = Parameter(torch.ones(1))

        self.var_adaptor = VarianceAdaptor(
            model_dim,
            decoder_frontend_config.subword_to_unit_upsampling_type,
            decoder_frontend_config.duration_predictor_config,
            decoder_frontend_config.pitch_predictor_config,
            decoder_frontend_config.energy_predictor_config,
        )

        layer_norm = decoder_frontend_config.layer_norm
        if layer_norm:
            self.layer_norm = StandardLayerNorm(model_dim, device=device, dtype=dtype)
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def subword_to_character_upsampling(self):
        pass

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        padding_mask = None
        # inference: subword_to_character_upsampling
        self.subword_to_character_upsampling()

        return seqs, padding_mask
