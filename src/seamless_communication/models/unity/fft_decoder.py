# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple, final

from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerNormOrder, create_standard_layer_norm
from fairseq2.typing import DataType, Device, finaloverride
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.unity.fft_decoder_layer import (
    FeedForwardTransformerLayer,
)


@final
class FeedForwardTransformer(Module):
    """Represents a Feedforward Transformer decoder."""

    model_dim: int
    layer_norm: Optional[LayerNorm]
    norm_order: TransformerNormOrder

    def __init__(
        self,
        layers: Iterable[FeedForwardTransformerLayer],
        *,
        norm_order: TransformerNormOrder = TransformerNormOrder.POST,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        :param norm_order:
            The Layer Normalization order to use.
        """
        super().__init__()

        layer_list = ModuleList(layers)

        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        self.model_dim = layer_list[0].model_dim

        self.layers = layer_list

        if norm_order != TransformerNormOrder.POST:
            self.layer_norm = create_standard_layer_norm(
                self.model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("layer_norm", None)

        self.norm_order = norm_order

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        for layer in self.layers.drop_iter():
            seqs, padding_mask = layer(seqs, padding_mask, film_cond_emb=film_cond_emb)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        return seqs, padding_mask

    def extra_repr(self) -> str:
        """:meta private:"""
        s = super().extra_repr()

        return f"{s}, norm_order={self.norm_order}"
