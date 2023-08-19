# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
import torch.nn as nn
from fairseq2.typing import Device

from seamless_communication.models.vocoder.codehifigan import CodeGenerator


class Vocoder(nn.Module):
    def __init__(self, code_generator: CodeGenerator, lang_spkr_idx_map: dict):
        super(Vocoder, self).__init__()
        self.code_generator = code_generator
        self.lang_spkr_idx_map = lang_spkr_idx_map

    def forward(
        self,
        code: List[int],
        lang: str,
        spkr: Optional[int] = -1,
        dur_prediction: bool = True,
    ):
        x = {
            "code": torch.LongTensor(code).view(1, -1),
        }
        lang_idx = self.lang_spkr_idx_map["multilingual"][lang]
        spkr_list = self.lang_spkr_idx_map["multispkr"][lang]
        if not spkr:
            spkr = -1
        spkr = spkr_list[0] if spkr == -1 else spkr
        x["spkr"] = torch.tensor([[spkr]])
        x["lang"] = torch.tensor([[lang_idx]])
        return self.code_generator(x, dur_prediction)
