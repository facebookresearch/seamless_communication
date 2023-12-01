# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, final

from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.projection import Projection
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.monotonic_decoder.monotonic_decoder import (
    MonotonicTransformerDecoder,
)


@final
class MonotonicDecoderModel(Module):
    text_decoder_frontend: TransformerFrontend
    text_decoder: MonotonicTransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        text_decoder_frontend: TransformerFrontend,
        text_decoder: MonotonicTransformerDecoder,
        final_proj: Projection,
    ) -> None:
        super().__init__()

        self.text_decoder_frontend = text_decoder_frontend
        self.text_decoder = text_decoder
        self.final_proj = final_proj

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        seqs, padding_mask = self.text_decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.text_decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    @finaloverride
    def project(self, decoder_output: Tensor) -> Tensor:
        logits = self.final_proj(decoder_output)

        return logits  # type: ignore[no-any-return]
