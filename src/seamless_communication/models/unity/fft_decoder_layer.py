# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask, apply_padding_mask
from fairseq2.nn.transformer import MultiheadAttention, create_standard_layer_norm
from fairseq2.typing import DataType, Device, finaloverride
from torch import Tensor
from torch.nn import Conv1d, Dropout, Module, ReLU

from seamless_communication.models.unity.film import FiLM


@final
class Conv1dBlock(Module):
    """Represents the Conv1d block within the FFT Block as described in
    :cite:t:`https://arxiv.org/pdf/1905.09263.pdf`."""

    conv1: Conv1d
    activation: ReLU
    conv2: Conv1d

    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        kernel_size: int,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param inner_dim:
            The inner dimensionality between the two convolutional layers.
        :param kernel_size:
            The kernel size of the Conv1d layers.
        :param bias:
            If ``True``, both the inner and output projections learn an additive
            bias.
        """
        super().__init__()

        self.conv1 = Conv1d(
            model_dim,
            inner_dim,
            kernel_size,
            stride=1,
            padding="same",
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.activation = ReLU()

        self.conv2 = Conv1d(
            inner_dim,
            model_dim,
            kernel_size,
            stride=1,
            padding="same",
            bias=bias,
            device=device,
            dtype=dtype,
        )

    @finaloverride
    def forward(self, seqs: Tensor, padding_mask: Optional[PaddingMask]) -> Tensor:
        # Ensure that we do not leak padded positions in the convolution layer.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        # (N, M, S) -> (N, inner_dim, S)
        seqs = self.conv1(seqs)

        # (N, inner_dim, S) -> (N, S, inner_dim)
        seqs = seqs.transpose(1, 2)

        seqs = apply_padding_mask(seqs, padding_mask)

        seqs = self.activation(seqs)

        # (N, S, inner_dim) -> (N, inner_dim, S)
        seqs = seqs.transpose(1, 2)

        # (N, inner_dim, S) -> (N, M, S)
        seqs = self.conv2(seqs)

        # (N, M, S) -> (N, S, M)
        seqs = seqs.transpose(1, 2)

        return seqs


@final
class FeedForwardTransformerLayer(Module):
    """Represents the FFT Block as described in
    :cite:t:`https://arxiv.org/pdf/1905.09263.pdf`."""

    model_dim: int
    self_attn: MultiheadAttention
    self_attn_dropout: Optional[Dropout]
    self_attn_layer_norm: LayerNorm
    conv1d: Conv1dBlock
    conv1d_dropout: Optional[Dropout]
    conv1d_layer_norm: LayerNorm
    film: Optional[FiLM]

    def __init__(
        self,
        self_attn: MultiheadAttention,
        conv1d: Conv1dBlock,
        dropout_p: float = 0.1,
        conv1d_dropout_p: float = 0.1,
        use_film: bool = False,
        film_cond_dim: int = 512,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param self_attn:
            The self attention layer.
        :param conv1d:
            The conv1d block.
        :param dropout_p:
            The dropout probability on the outputs of the self attention layer.
        :param conv1d_dropout_p:
            The dropout probability on the outputs of the conv1d block.
        :param use_film:
            Whether to condition on a fixed-size vector through FiLM.
        :param film_cond_dim:
            The dim of fixed-size vector conditioned on during model forward.
        """
        super().__init__()

        self.model_dim = self_attn.model_dim

        self.self_attn = self_attn

        if dropout_p > 0.0:
            self.self_attn_dropout = Dropout(dropout_p)
        else:
            self.register_module("self_attn_dropout", None)

        layer_norm_factory = create_standard_layer_norm

        self.self_attn_layer_norm = layer_norm_factory(
            self.model_dim, device=device, dtype=dtype
        )

        self.conv1d = conv1d

        if conv1d_dropout_p > 0.0:
            self.conv1d_dropout = Dropout(conv1d_dropout_p)
        else:
            self.register_module("conv1d_dropout", None)

        self.conv1d_layer_norm = layer_norm_factory(
            self.model_dim, device=device, dtype=dtype
        )

        if use_film:
            self.film = FiLM(film_cond_dim, self.model_dim, device=device, dtype=dtype)
        else:
            self.register_module("film", None)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs = self._forward_self_attn(seqs, padding_mask)

        seqs = self._forward_conv1d(seqs, padding_mask)

        if self.film is not None and film_cond_emb is not None:
            seqs = self.film(seqs, film_cond_emb)
            seqs = apply_padding_mask(seqs, padding_mask)

        return seqs, padding_mask

    def _forward_self_attn(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
    ) -> Tensor:
        residual = seqs

        seqs = self.self_attn(
            seqs,
            padding_mask,
            keys=seqs,
            key_padding_mask=padding_mask,
            values=seqs,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual

        seqs = self.self_attn_layer_norm(seqs)

        return seqs

    def _forward_conv1d(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tensor:
        residual = seqs

        seqs = self.conv1d(seqs, padding_mask)

        if self.conv1d_dropout is not None:
            seqs = self.conv1d_dropout(seqs)

        seqs = seqs + residual

        seqs = self.conv1d_layer_norm(seqs)

        return seqs
