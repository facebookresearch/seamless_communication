# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from fairseq2.nn.transformer import AttentionMask, CustomAttentionMask
from fairseq2.nn.utils.mask import to_float_mask
from torch import Tensor


class ChunkAttentionMaskFactory:
    """Generates a chunk attention mask for self attention.

    .. note::
        This class follows the :class:`AttentionMaskGenerator` protocol.
    """

    def __init__(
        self, chunk_size: int, left_chunk_num: int, right_chunk_num: int
    ) -> None:
        self.chunk_size = chunk_size
        self.left_chunk_num = left_chunk_num
        self.right_chunk_num = right_chunk_num

        if self.right_chunk_num != 0:
            raise ValueError("We currently only support `right_chunk_num` == 0.")

    def __call__(self, seqs: Tensor) -> Optional[AttentionMask]:
        """
        :param seqs:
            The sequences for which to generate the mask. *Shape:*
            :math:`(N,S,M)`, where :math:`N` is the batch size, :math:`S` is the
            sequence length, and :math:`M` is the dimensionality of the model.

        :returns:
            A chunk attention float mask for ``seqs``.
            *Shape:* :math:`(S,S)`, where :math:`S` is the
            sequence length.
        """

        seq_len = seqs.size(1)

        chunk_indices = torch.div(
            torch.arange(seq_len, device=seqs.device), self.chunk_size
        ).long()

        start_indices = (
            ((chunk_indices - self.left_chunk_num).clamp_(min=0) * self.chunk_size).to(
                seqs.device
            )
            if self.left_chunk_num >= 0
            else torch.full_like(chunk_indices, 0)
        )
        start_indices = start_indices.unsqueeze(1).expand(-1, seq_len)

        end_indices = (
            ((chunk_indices + 1) * self.chunk_size).clamp_(max=seq_len).to(seqs.device)
        )

        end_indices = end_indices.unsqueeze(1).expand(-1, seq_len)

        indices = (
            torch.arange(seq_len, device=seqs.device).unsqueeze(0).expand(seq_len, -1)
        )

        bool_mask = (indices < start_indices) | (indices >= end_indices)

        mask = to_float_mask(bool_mask, seqs.dtype)

        mask = mask[:seq_len, :seq_len]

        return CustomAttentionMask(mask)
