# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    AttentionMask,
    FeedForwardNetwork,
    MultiheadAttention,
    create_standard_layer_norm,
)
from fairseq2.typing import DataType, Device, finaloverride
from torch import Tensor
from torch.nn import Dropout, Module

from seamless_communication.models.monotonic_decoder.p_choose import PChooseLayer


@final
class MonotonicTransformerDecoderLayer(Module):
    """Represents a Monotonic Transformer decoder layer."""

    self_attn: MultiheadAttention
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    encoder_decoder_attn: MultiheadAttention
    encoder_decoder_attn_dropout: Optional[Dropout]
    encoder_decoder_attn_layer_norm: LayerNorm
    p_choose_layer: PChooseLayer
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]
    ffn_layer_norm: LayerNorm

    def __init__(
        self,
        self_attn: MultiheadAttention,
        encoder_decoder_attn: MultiheadAttention,
        p_choose_layer: PChooseLayer,
        ffn: FeedForwardNetwork,
        *,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param encoder_decoder_attn:
            The encoder-decoder attention layer.
        :param ffn:
            The feed-forward network.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
        """
        super().__init__()

        self.model_dim = self_attn.model_dim

        self_attn_layer_norm = create_standard_layer_norm(
            self.model_dim, device=device, dtype=dtype
        )

        self.self_attn_layer_norm = self_attn_layer_norm

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        encoder_decoder_attn_layer_norm = create_standard_layer_norm(
            self.model_dim, device=device, dtype=dtype
        )

        self.encoder_decoder_attn_layer_norm = encoder_decoder_attn_layer_norm

        self.encoder_decoder_attn = encoder_decoder_attn

        if dropout_p > 0.0:
            self.encoder_decoder_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("encoder_decoder_attn_dropout", None)

        self.p_choose_layer = p_choose_layer

        ffn_layer_norm = create_standard_layer_norm(
            self.model_dim, device=device, dtype=dtype
        )

        self.ffn_layer_norm = ffn_layer_norm

        self.ffn = ffn

        if dropout_p > 0.0:
            self.ffn_dropout = Dropout(dropout_p)
        else:
            self.register_module("ffn_dropout", None)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask, state_bag)

        seqs, p_choose = self._forward_encoder_decoder_attn(
            seqs, padding_mask, encoder_output, encoder_padding_mask
        )

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask, p_choose

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn_layer_norm(seqs)

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
            state_bag=state_bag,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        return seqs

    def _forward_encoder_decoder_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Optional[Tensor],
        encoder_padding_mask: Optional[PaddingMask],
    ) -> Tuple[Tensor, Tensor]:
        if encoder_output is None:
            raise ValueError(
                "`encoder_output` must not be `None` for encoder-decoder attention."
            )

        residual = seqs

        seqs = self.encoder_decoder_attn_layer_norm(seqs)

        p_choose = self.p_choose_layer(seqs, encoder_output)

        seqs = self.encoder_decoder_attn(
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            encoder_output,
        )

        if self.encoder_decoder_attn_dropout is not None:
            seqs = self.encoder_decoder_attn_dropout(seqs)

        seqs = seqs + residual

        return seqs, p_choose

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        seqs = seqs + residual

        return seqs
