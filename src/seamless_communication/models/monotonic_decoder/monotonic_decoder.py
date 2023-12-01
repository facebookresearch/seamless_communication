# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Iterable, List, Optional, Tuple, final

import torch
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import (
    AttentionMaskFactory,
    CausalAttentionMaskFactory,
    create_standard_layer_norm,
)
from fairseq2.typing import DataType, Device, finaloverride
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.monotonic_decoder.monotonic_decoder_layer import (
    MonotonicTransformerDecoderLayer,
)


@final
class MonotonicTransformerDecoder(Module):
    """Represents a Monotonic Transformer decoder."""

    model_dim: int
    self_attn_mask_factory: AttentionMaskFactory
    layers: ModuleList
    layer_norm: LayerNorm

    def __init__(
        self,
        layers: Iterable[MonotonicTransformerDecoderLayer],
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param layers:
            The decoder layers.
        """
        super().__init__()

        layer_list = ModuleList(layers)

        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        self.model_dim = layer_list[0].model_dim

        self.self_attn_mask_factory = CausalAttentionMaskFactory()

        self.layers = layer_list

        self.layer_norm = create_standard_layer_norm(
            self.model_dim, device=device, dtype=dtype
        )

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Optional[Tensor] = None,
        encoder_padding_mask: Optional[PaddingMask] = None,
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        self_attn_mask = self.self_attn_mask_factory(
            seqs, keys=seqs, training=self.training, state_bag=state_bag
        )

        p_choose_list: List[Tensor] = []

        for layer in self.layers.drop_iter():
            seqs, padding_mask, p_choose = layer(
                seqs,
                padding_mask,
                self_attn_mask,
                encoder_output,
                encoder_padding_mask,
                state_bag=state_bag,
            )
            p_choose_list.append(p_choose)

        seqs = self.layer_norm(seqs)

        p_choose = torch.cat(p_choose_list, dim=0)

        p_choose = p_choose.flatten(0, 1)

        return seqs, padding_mask, p_choose
