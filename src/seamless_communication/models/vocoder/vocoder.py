# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.vocoder.codehifigan import CodeGenerator


class Vocoder(Module):
    def __init__(
        self,
        code_generator: CodeGenerator,
        lang_spkr_idx_map: Dict[str, Any],
    ):
        super().__init__()
        self.code_generator = code_generator
        self.lang_spkr_idx_map = lang_spkr_idx_map

    def forward(
        self,
        units: Tensor,
        lang: str,
        spkr: Optional[int] = -1,
        dur_prediction: bool = True,
    ) -> Tensor:
        lang_idx = self.lang_spkr_idx_map["multilingual"][lang]
        spkr_list = self.lang_spkr_idx_map["multispkr"][lang]
        if not spkr:
            spkr = -1
        spkr = spkr_list[0] if spkr == -1 else spkr
        x = {
            "code": units.view(1, -1),
            "spkr": torch.tensor([[spkr]], device=units.device),
            "lang": torch.tensor([[lang_idx]], device=units.device),
        }
        return self.code_generator(x, dur_prediction)  # type: ignore[no-any-return]
