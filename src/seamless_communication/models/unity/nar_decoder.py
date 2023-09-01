# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, final, Tuple

from torch import Tensor
from torch.nn import Conv1d, Dropout, Module, ReLU

from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.transformer import (
    create_default_layer_norm,
    LayerNorm,
    TransformerDecoderLayer,
    MultiheadAttention,
)
from fairseq2.typing import DataType, Device, finaloverride

from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.utils.module import check_model_dim
from fairseq2.typing import DataType, Device, finaloverride


@final
class Conv1dBlock(Module):
    """Represents the Conv1d block within the FFT Block as described in
    :cite:t:`https://arxiv.org/pdf/1905.09263.pdf`."""

    inner_layer: Conv1d
    inner_dropout: Optional[Dropout]
    inner_norm: Optional[LayerNorm]
    outer_layer: Conv1d

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        kernel_size: int,
        inner_dropout_p: float = 0.0,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The dimensionality of the inner projection layer.
        :param kernel_size:
            The kernel size of the Conv1d layers.
        :param inner_activation:
            The activation to apply to outputs of the inner projection layer. If
            ``None``, :func:`~torch.nn.ReLU` will be used.
        :param inner_dropout_p:
            The dropout probability on outputs of the inner projection layer.
        :param bias:
            If ``True``, both the inner and output projections learn an additive
            bias.
        :param layer_norm_fn:
            The factory to use to construct the Layer Normalization module.
        """
        super().__init__(model_dim)

        self.conv1 = Conv1d(
            model_dim,
            inner_dim,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.inner_activation = ReLU()

        self.conv2 = Conv1d(
            inner_dim,
            model_dim,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        if inner_dropout_p > 0.0:
            self.inner_dropout = Dropout(inner_dropout_p)
        else:
            self.register_module("inner_dropout", None)

        self.layer_norm = create_default_layer_norm(inner_dim, device, dtype)

    @finaloverride
    def forward(self, seqs: Tensor) -> Tensor:
        # TODO: Should I transpose seqs?
        residual = seqs

        seqs = self.conv1(seqs)

        seqs = self.inner_activation(seqs)

        seqs = self.conv2(seqs)

        if self.inner_dropout is not None:
            seqs = self.inner_dropout(seqs)

        seqs = seqs + residual

        seqs = self.layer_norm(seqs)

        return seqs


@final
class NARTransformerDecoderLayer(TransformerDecoderLayer):
    """Represents a Transformer decoder layer as described in
    :cite:t:`https://doi.org/10.48550/arxiv.1706.03762`."""

    self_attn: MultiheadAttention
    self_attn_norm: Optional[LayerNorm]
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    conv1d_block: Conv1dBlock
    conv1d_dropout: Optional[Dropout]
    conv1d_layer_norm: LayerNorm

    def __init__(
        self,
        self_attn: MultiheadAttention,
        conv1d_block: Conv1dBlock,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param ffn:
            The feed-forward network.
        :param dropout_p:
            The dropout probability on outputs of the attention layers and the
            feed-forward network.
        :param layer_norm_fn:
            The factory to use to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_fn is None:
            layer_norm_fn = create_default_layer_norm

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        self.self_attn_layer_norm = layer_norm_fn(model_dim, device, dtype)

        self.conv1d_block = conv1d_block

        if dropout_p > 0.0:
            self.conv1d_dropout = Dropout(dropout_p)
        else:
            self.register_module("conv1d_dropout", None)

        conv1d_layer_norm = layer_norm_fn(model_dim, device, dtype)

        self.conv1d_layer_norm = conv1d_layer_norm

        check_model_dim(self)

        self.reset_parameters()

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor] = None,
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[Tensor] = None,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs = self._forward_self_attn(seqs, padding_mask, self_attn_mask, state_bag)

        seqs = self._forward_conv1d(
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            values=seqs,
            attn_mask=self_attn_mask,
            key_padding_mask=padding_mask,
            state_bag=state_bag,
        )

        if self.self_attn_norm is not None:
            seqs = self.self_attn_norm(seqs)

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_conv1d(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        self_attn_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag],
    ) -> Tensor:
        return
