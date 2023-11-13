# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from seamless_communication.models.vocoder.hifigan import Generator


class MelGenerator(Generator):
    def __init__(
        self,
        upsample_rates: List[int],
        upsample_kernel_sizes: List[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        model_in_dim: int = 80,
    ):
        super().__init__(
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            model_in_dim,
            add_ups_out_pad=True,
        )

        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            assert k == 2 * u, (k, u)

        mean = torch.zeros((model_in_dim,), dtype=torch.float)
        scale = torch.zeros((model_in_dim,), dtype=torch.float)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor, normalize_before: bool = True) -> torch.Tensor:
        if normalize_before:
            x = (x - self.mean) / self.scale
        x = super().forward(x.transpose(1, 0).unsqueeze(0))
        return x.squeeze(0).transpose(1, 0)
