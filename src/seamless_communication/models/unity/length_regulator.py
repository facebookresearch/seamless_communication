# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from torch.nn import Module


class GaussianUpsampling(Module):
    """Gaussian upsampling with fixed temperature as in
    :cite:t:`https://arxiv.org/abs/2010.04301`."""

    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(self):
        pass


class HardUpsampling(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        # Stretch the current tensor as per durations tensor.
        pass
