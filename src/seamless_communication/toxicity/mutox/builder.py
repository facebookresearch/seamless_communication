# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import typing as tp
from seamless_communication.toxicity.mutox.classifier import (
    MutoxClassifier,
    MutoxConfig,
)
import torch
from torch import nn
from fairseq2.typing import DataType, Device


class MutoxClassifierBuilder:
    """
    Builder module for MutoxClassifier model
    """

    config: MutoxConfig
    device: tp.Optional[Device]
    dtype: tp.Optional[DataType]

    def __init__(
        self,
        config: MutoxConfig,
        *,
        device: tp.Optional[Device] = None,
        dtype: tp.Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param device:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config
        self.device, self.dtype = device, dtype

    def build_model(self) -> MutoxClassifier:
        model_h1 = nn.Sequential(
            nn.Dropout(0.01),
            nn.Linear(self.config.input_size, 512),
        )

        model_h2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 128),
        )

        model_h3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        model_all = nn.Sequential(
            model_h1,
            model_h2,
            model_h3,
        )

        return MutoxClassifier(model_all,).to(
            device=self.device,
            dtype=self.dtype,
        )


def create_mutox_model(
    config: MutoxConfig,
    device: tp.Optional[Device] = None,
    dtype: tp.Optional[DataType] = None,
) -> MutoxClassifier:
    """Create a Mutox Classifier model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    return MutoxClassifierBuilder(
        config,
        device=device,
        dtype=dtype,
    ).build_model()
