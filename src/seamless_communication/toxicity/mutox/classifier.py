# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import torch
from torch import nn
from fairseq2.typing import DataType, Device

from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from typing import Optional


class MutoxClassifier(nn.Module):
    def __init__(
        self,
        model_all,
    ):
        super().__init__()
        self.model_all = model_all

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model_all(inputs)


@dataclass
class MutoxConfig:
    """Holds the configuration of a Mutox Classifier model."""

    # size of the input embedding supported by this model
    input_size: int


mutox_archs = ArchitectureRegistry[MutoxConfig]("mutox_classifier")
