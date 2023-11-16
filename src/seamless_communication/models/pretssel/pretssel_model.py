# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.nn.embedding import Embedding, StandardEmbedding
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Projection
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Conv1d,
    Dropout,
    Module,
    ModuleList,
    Parameter,
    Sequential,
    Tanh,
    init,
)

from seamless_communication.models.pretssel.ecapa_tdnn import ECAPA_TDNN
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.length_regulator import VarianceAdaptor


class PretsselEncoderFrontend(Module):
    """Represent Encoder frontend, including speaker & language embedding"""

    prosody_encoder: ECAPA_TDNN
    embed_tokens: Embedding
    embed_positions: PositionEncoder
    pos_emb_alpha: Parameter
    embed_lang: Embedding
    dropout: Dropout

    def __init__(
        self,
        prosody_encoder: ECAPA_TDNN,
        embed_tokens: Embedding,
        embed_positions: PositionEncoder,
        lang_to_index: Dict[str, int],
        lang_embed_dim: Optional[int],
        dropout_p: float,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.prosody_encoder = prosody_encoder

        self.embed_tokens = embed_tokens

        self.embed_positions = embed_positions
        self.pos_emb_alpha = Parameter(torch.ones(1, device=device, dtype=dtype))

        self.lang_to_index = lang_to_index

        if lang_embed_dim is not None:
            self.embed_lang = StandardEmbedding(
                len(lang_to_index), lang_embed_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("embed_lang", None)

        self.dropout = Dropout(dropout_p)

        self.device = device
        self.dtype = dtype

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        prosody_input_seqs: Tensor,
        prosody_padding_mask: Optional[PaddingMask],
        tgt_lang: str,
    ) -> Tuple[Tensor, Tensor]:
        prosody_embs = self.prosody_encoder(
            prosody_input_seqs,
            prosody_padding_mask,
        ).unsqueeze(1)

        if self.embed_lang is not None:
            lang_index = self.lang_to_index[tgt_lang]
            lang_index_tensor = (
                torch.tensor([lang_index]).to(seqs).repeat(seqs.size(0), 1)
            )
            lang_embeds = self.embed_lang(lang_index_tensor)
            prosody_embs = torch.cat([prosody_embs, lang_embeds], dim=-1)

        seqs = self.embed_tokens(seqs)
        seqs += self.pos_emb_alpha * (self.embed_positions(seqs, padding_mask) - seqs)
        seqs = self.dropout(seqs)

        return seqs, prosody_embs


class PretsselDecoderFrontend(Module):
    """Represent Decoder frontend, including VarianceAdaptor & Positional embedding"""

    variance_adaptor: VarianceAdaptor
    embed_positions: PositionEncoder
    pos_emb_alpha: Parameter

    def __init__(
        self,
        variance_adaptor: VarianceAdaptor,
        embed_positions: PositionEncoder,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.variance_adaptor = variance_adaptor
        self.embed_positions = embed_positions
        self.pos_emb_alpha = Parameter(torch.ones(1, device=device, dtype=dtype))

        self.device = device
        self.dtype = dtype

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        durations: Optional[Tensor] = None,
        duration_factor: float = 1.0,
        min_duration: int = 0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.variance_adaptor(
            seqs, padding_mask, durations, duration_factor, min_duration, film_cond_emb
        )

        seqs += self.pos_emb_alpha * (self.embed_positions(seqs, padding_mask) - seqs)

        return seqs, padding_mask


class PostNet(Module):
    """Represent a PostNet"""

    def __init__(
        self,
        in_dim: int,
        n_channels: int,
        kernel_size: int,
        n_layers: int,
        dropout: float,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.convolutions = ModuleList()
        assert kernel_size % 2 == 1
        for i in range(n_layers):
            cur_layers = (
                [
                    Conv1d(
                        in_dim if i == 0 else n_channels,
                        n_channels if i < n_layers - 1 else in_dim,
                        kernel_size=kernel_size,
                        padding="same",
                        device=device,
                        dtype=dtype,
                    ),
                    BatchNorm1d(
                        n_channels if i < n_layers - 1 else in_dim,
                        device=device,
                        dtype=dtype,
                    ),
                ]
                + ([Tanh()] if i < n_layers - 1 else [])
                + [Dropout(dropout)]
            )
            self.convolutions.append(Sequential(*cur_layers))

        self.device = device
        self.dtype = dtype
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""
        for i, layer in enumerate(self.convolutions):
            init.xavier_uniform_(
                layer[0].weight,
                init.calculate_gain(
                    "tanh" if i < len(self.convolutions) - 1 else "linear"
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)  # B x T x C -> B x C x T
        for layer in self.convolutions:
            x = layer(x)

        return x.transpose(1, 2)


class PretsselModel(Module):
    """Represent the PretsselModel"""

    encoder_frontend: PretsselEncoderFrontend
    encoder: FeedForwardTransformer
    decoder_frontend: PretsselDecoderFrontend
    decoder: FeedForwardTransformer
    final_proj: Projection
    postnet: PostNet

    def __init__(
        self,
        encoder_frontend: PretsselEncoderFrontend,
        encoder: FeedForwardTransformer,
        decoder_frontend: PretsselDecoderFrontend,
        decoder: FeedForwardTransformer,
        final_proj: Projection,
        postnet: PostNet,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.postnet = postnet

        self.device = device
        self.dtype = dtype

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        prosody_input_seqs: Tensor,
        prosody_padding_mask: Optional[PaddingMask],
        tgt_lang: str,
        durations: Optional[Tensor] = None,
        duration_factor: float = 1.0,
        min_duration: int = 0,
    ) -> Tensor:
        # (N, S) -> (N, S, M)
        seqs, cond_embs = self.encoder_frontend(
            seqs,
            padding_mask,
            prosody_input_seqs,
            prosody_padding_mask,
            tgt_lang,
        )

        seqs, padding_mask = self.encoder(seqs, padding_mask, cond_embs)

        # (N, S, M) -> (N, X, M), inflated units
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, durations, duration_factor, min_duration, cond_embs
        )

        seqs, padding_mask = self.decoder(seqs, padding_mask, cond_embs)

        # (N, X, M) -> (N, X, n_mels)
        seqs = self.final_proj(seqs)

        seqs = seqs + self.postnet(seqs)

        return seqs
