# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from fairseq2.nn.projection import Linear
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Module, Parameter


class FiLM(Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    proj: Linear
    s_gamma: Parameter
    s_beta: Parameter

    def __init__(
        self,
        cond_dim: int,
        embed_dim: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.proj = Linear(
            cond_dim, 2 * embed_dim, bias=True, device=device, dtype=dtype
        )

        self.s_gamma = Parameter(
            torch.ones(
                1,
                device=device,
                dtype=dtype,
            ),
            requires_grad=True,
        )

        self.s_beta = Parameter(
            torch.ones(
                1,
                device=device,
                dtype=dtype,
            ),
            requires_grad=True,
        )

    def forward(self, x: Tensor, cond_embs: Tensor) -> Tensor:
        """
        x -- [B, T, H]
        cond_emb -- [B, 1, C]
        """
        # get trainable gamma, beta
        gammas, betas = self.proj(cond_embs).chunk(2, dim=-1)  # B x 1 x H

        # apply film
        gammas = self.s_gamma * gammas.expand_as(x)
        betas = self.s_beta * betas.expand_as(x)

        return (gammas + 1.0) * x + betas  # type: ignore[no-any-return]
