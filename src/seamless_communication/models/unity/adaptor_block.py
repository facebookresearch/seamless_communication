# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple, final

import torch
from fairseq2.models.conformer import ConformerBlock
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import (
    AttentionMask,
    FeedForwardNetwork,
    LayerNormFactory,
    MultiheadAttention,
    TransformerEncoder,
    TransformerEncoderLayer,
    create_standard_layer_norm,
)
from fairseq2.typing import DataType, Device
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import GLU, Conv1d, Dropout, ReLU


@final
class UnitYEncoderAdaptor(TransformerEncoder):
    """Represents a Transformer encoder that wraps a speech encoder and adapts
    it to be used with the UnitY architecture."""

    inner: TransformerEncoder
    inner_layer_norm: Optional[LayerNorm]
    proj1: Linear
    activation: ReLU
    proj2: Linear
    adaptor_layers: ModuleList
    layer_norm: LayerNorm

    def __init__(
        self,
        inner: TransformerEncoder,
        adaptor_layers: Iterable[TransformerEncoderLayer],
        *,
        inner_layer_norm: bool = False,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param inner:
            The speech encoder to wrap.
        :param adaptor_layers:
            The adaptor layers to stack on top of ``inner``.
        :param inner_layer_norm:
            If ``True``, applies Layer Normalization to outputs of ``inner``.
        :param layer_norm_factory:
            The factory to use to construct the Layer Normalization modules.
        """
        model_dim = inner.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.inner = inner

        if inner_layer_norm:
            self.inner_layer_norm = layer_norm_factory(
                model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("inner_layer_norm", None)

        self.proj1 = Linear(
            model_dim, model_dim * 4, bias=True, device=device, dtype=dtype
        )

        self.activation = ReLU()

        self.proj2 = Linear(
            model_dim * 4, model_dim, bias=True, device=device, dtype=dtype
        )

        layer_list = ModuleList(adaptor_layers)
        if not layer_list:
            raise ValueError("`adaptor_layers` must be non-empty.")

        self.adaptor_layers = layer_list

        self.layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.inner(seqs, padding_mask)

        if self.inner_layer_norm is not None:
            seqs = self.inner_layer_norm(seqs)

        # Only difference compared to a vanilla Transformer encoder.
        seqs = seqs + 0.5 * self._expand_contract(seqs)

        for layer in self.adaptor_layers:
            seqs, padding_mask = layer(seqs, padding_mask)

        seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def _expand_contract(self, seqs: Tensor) -> Tensor:
        seqs = self.proj1(seqs)

        seqs = self.activation(seqs)

        seqs = self.proj2(seqs)

        return seqs


@final
class UnitYTransformerAdaptorLayer(TransformerEncoderLayer):
    """Represents a variant of M-Adaptor layer described in
    :cite:t`https://doi.org/10.48550/arxiv.2207.00952`.

    The main difference from the paper is that pooling is applied to multi-head
    attention input rather than projected Q, K, V.
    """

    kernel_size: int
    stride: int
    residual_layer_norm: LayerNorm
    residual_conv: Conv1d
    residual_activation: GLU
    self_attn_layer_norm: LayerNorm
    self_attn_conv: Conv1d
    self_attn_activation: GLU
    self_attn: MultiheadAttention
    self_attn_dropout: Optional[Dropout]
    ffn_layer_norm: LayerNorm
    ffn: FeedForwardNetwork
    ffn_dropout: Optional[Dropout]

    def __init__(
        self,
        self_attn: MultiheadAttention,
        ffn: FeedForwardNetwork,
        kernel_size: int,
        stride: int,
        *,
        dropout_p: float = 0.1,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param ffn:
            The feed-forward network.
        :param kernel_size:
            The kernel size for 1D pooling convolutions.
        :param stride:
            The stride for 1D pooling convolutions.
        :param dropout_p:
            The dropout probability on outputs of the self attention layer and
            the feed-forward network.
        :param layer_norm_factory:
            The factory to use to construct the Layer Normalization modules.
        """
        model_dim = self_attn.model_dim

        super().__init__(model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.kernel_size = kernel_size
        self.stride = stride

        self.residual_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        self.residual_conv = Conv1d(
            model_dim,
            model_dim * 2,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            device=device,
            dtype=dtype,
        )

        self.residual_activation = GLU(dim=1)

        self.self_attn_layer_norm = layer_norm_factory(
            model_dim, device=device, dtype=dtype
        )

        self.self_attn_conv = Conv1d(
            model_dim,
            model_dim * 2,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            device=device,
            dtype=dtype,
        )

        self.self_attn_activation = GLU(dim=1)

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        self.ffn_layer_norm = layer_norm_factory(model_dim, device=device, dtype=dtype)

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
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self._forward_self_attn(seqs, padding_mask, self_attn_mask)

        seqs = self._forward_ffn(seqs)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask],
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        residual = self.residual_layer_norm(seqs)

        # Apply pooling to the residual to match the sequence length of the
        # multi-head attention output.
        # (N, S, M) -> (N, M, S)
        residual = residual.transpose(1, 2)

        residual = self.residual_conv(residual)

        residual = self.residual_activation(residual)

        # (N, M, S) -> (N, S, M)
        residual = residual.transpose(1, 2)

        seqs = self.self_attn_layer_norm(seqs)

        # Apply pooling before feeding to the multihead-attention layer.
        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        seqs = self.self_attn_conv(seqs)

        seqs = self.self_attn_activation(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        padding_mask = _compute_new_padding_mask(
            seqs, padding_mask, self.kernel_size, self.stride
        )

        # The rest of the computation is identical to a vanilla Transformer
        # encoder layer.
        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
            attn_mask=self_attn_mask,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        return seqs, padding_mask

    def _forward_ffn(self, seqs: Tensor) -> Tensor:
        residual = seqs

        seqs = self.ffn_layer_norm(seqs)

        seqs = self.ffn(seqs)

        if self.ffn_dropout is not None:
            seqs = self.ffn_dropout(seqs)

        return seqs + residual

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", kernel_size={self.kernel_size}, stride={self.stride}"


@final
class UnitYConformerAdaptorLayer(TransformerEncoderLayer):
    """Represents a variant of M-Adaptor layer described in
    :cite:t`https://doi.org/10.48550/arxiv.2207.00952`.

    The main difference from the paper is that this variant uses a Conformer
    block which empirically showed better performance when used with Conformer-
    based speech encoder architectures such as w2v-BERT.
    """

    kernel_size: int
    stride: int
    layer_norm: Optional[LayerNorm]
    conv: Conv1d
    activation: GLU
    block: ConformerBlock

    def __init__(
        self,
        block: ConformerBlock,
        kernel_size: int,
        stride: int,
        *,
        layer_norm: bool = False,
        layer_norm_factory: Optional[LayerNormFactory] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param block:
            The Conformer block to use.
        :param kernel_size:
            The kernel size for 1D pooling convolutions.
        :param stride:
            The stride for 1D pooling convolutions.
        :param layer_norm:
            If ``True``, applies Layer Normalization to inputs before pooling.
        :param layer_norm_factory:
            The factory to use to construct the Layer Normalization modules.
        """
        super().__init__(block.model_dim)

        if layer_norm_factory is None:
            layer_norm_factory = create_standard_layer_norm

        self.kernel_size = kernel_size
        self.stride = stride

        if layer_norm:
            self.layer_norm = layer_norm_factory(
                self.model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("layer_norm", None)

        self.conv = Conv1d(
            self.model_dim,
            self.model_dim * 2,
            kernel_size,
            stride,
            padding=kernel_size // 2,
            device=device,
            dtype=dtype,
        )

        self.activation = GLU(dim=1)

        self.block = block

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        self_attn_mask: Optional[AttentionMask] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        # Apply pooling before feeding to the Conformer block.
        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        seqs = self.conv(seqs)

        seqs = self.activation(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        padding_mask = _compute_new_padding_mask(
            seqs, padding_mask, self.kernel_size, self.stride
        )

        return self.block(seqs, padding_mask, self_attn_mask)  # type: ignore[no-any-return]

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return s + f", kernel_size={self.kernel_size}, stride={self.stride}"


def _compute_new_padding_mask(
    seqs: Tensor, padding_mask: Optional[PaddingMask], kernel_size: int, stride: int
) -> Optional[PaddingMask]:
    if padding_mask is None:
        return padding_mask

    pad = kernel_size // 2

    seq_lens = ((padding_mask.seq_lens + 2 * pad - kernel_size) / stride) + 1

    seq_lens = seq_lens.floor().to(torch.int64)

    return PaddingMask(seq_lens, batch_seq_len=seqs.size(1))
