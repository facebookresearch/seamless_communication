# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, List, Union
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
        lang_list: Union[List[str], str],
        spkr_list: Union[Optional[List[int]], int] = None,
        dur_prediction: bool = True,
    ) -> Tensor:
        # TODO: Do we need this backward compatibility, or just update all calling sites? 
        if len(units.shape) == 1:
            units = units.unsqueeze(0) # add batch dim
        if isinstance(lang_list, str):
            lang_list = [lang_list] * units.size(0)
        if isinstance(spkr_list, int):
            spkr_list = [spkr_list] * units.size(0)
        lang_idx_list = [self.lang_spkr_idx_map["multilingual"][l] for l in lang_list]
        if not spkr_list:
            spkr_list = [-1 for _ in range(len(lang_list))]
        spkr_list = [self.lang_spkr_idx_map["multispkr"][lang_list[i]][0] if spkr_list[i] == -1 else spkr_list[i] for i in range(len(spkr_list))]
        x = {
            "code": units.view(units.size(0), -1),
            "spkr": torch.tensor([spkr_list], device=units.device).t(),
            "lang": torch.tensor([lang_idx_list], device=units.device).t(),

        }
        return self.code_generator(x, dur_prediction)  # type: ignore[no-any-return]
