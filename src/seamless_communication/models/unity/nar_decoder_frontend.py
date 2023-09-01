# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from torch import Tensor
from torch.nn import Dropout

from fairseq2.models.transformer import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.typing import DataType, Device, finaloverride


@final
class NARDecoderFrontend(TransformerFrontend):
    """Represents a NAR Decoder front-end."""

    def __init__(
        self,
        model_dim: int,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        super().__init__(model_dim)

    @finaloverride
    def forward(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        padding_mask = None
        return seqs, padding_mask
