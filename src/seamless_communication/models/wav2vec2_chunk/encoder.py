# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Iterable, Optional, Tuple, final

from fairseq2.nn.module_list import ModuleList
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer import TransformerEncoder, TransformerEncoderLayer
from fairseq2.typing import finaloverride
from torch import Tensor
from torch.nn import Dropout

from seamless_communication.models.wav2vec2_chunk.chunk_attention_mask import (
    ChunkAttentionMaskFactory,
)


@final
class ChunkTransformerEncoder(TransformerEncoder):
    """Represents a Chunk Transformer encoder."""

    preliminary_dropout: Optional[Dropout]
    self_attn_mask_factory: ChunkAttentionMaskFactory
    layers: ModuleList
    layer_norm: Optional[LayerNorm]

    def __init__(
        self,
        layers: Iterable[TransformerEncoderLayer],
        chunk_size: int,
        left_chunk_num: int,
        right_chunk_num: int,
        *,
        dropout_p: float = 0.0,
        layer_drop_p: float = 0.0,
    ) -> None:
        """
        :param layers:
            The encoder layers.
        :param chunk_size:
            Size of each chunk.
        :param left_chunk_num:
            Number of chunks on the left up to which lookahead is allowed.
        :param right_chunk_num:
            Number of chunks on the right up to which lookahead is allowed.
        :param dropout_p:
            Used in the preliminary dropout.
        :param layer_drop_p:
            If greater than zero, applies LayerDrop to the encoder layers as
            described in :cite:t:`https://doi.org/10.48550/arxiv.1909.11556`.
        """
        layer_list = ModuleList(layers, drop_p=layer_drop_p)
        if not layer_list:
            raise ValueError("`layers` must be non-empty.")

        model_dim = layer_list[0].model_dim

        super().__init__(model_dim)

        if dropout_p > 0.0:
            self.preliminary_dropout = Dropout(dropout_p)
        else:
            self.register_module("preliminary_dropout", None)

        self.self_attn_mask_factory = ChunkAttentionMaskFactory(
            chunk_size * 2, left_chunk_num, right_chunk_num
        )

        self.layers = layer_list

    @finaloverride
    def forward(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self._layer_output_hooks and self.layers.drop_p > 0.0:
            raise ValueError(
                "The layer output hooks cannot be run when LayerDrop is enabled."
            )

        if self.preliminary_dropout is not None:
            seqs = self.preliminary_dropout(seqs)

        self_attn_mask = self.self_attn_mask_factory(seqs)

        num_layers = len(self.layers)

        for layer_idx, layer in enumerate(self.layers.drop_iter()):
            seqs, padding_mask = layer(seqs, padding_mask, self_attn_mask)

            for hook in self._layer_output_hooks.values():
                if not hook(layer_idx, seqs, padding_mask, num_layers):
                    break

        return seqs, padding_mask
