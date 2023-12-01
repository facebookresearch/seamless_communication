# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional

from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.typing import DataType, Device

from seamless_communication.models.generator.ecapa_tdnn import ECAPA_TDNN


@dataclass
class EcapaTDNNConfig:
    channels: List[int]
    kernel_sizes: List[int]
    dilations: List[int]
    attention_channels: int
    res2net_scale: int
    se_channels: int
    global_context: bool
    groups: List[int]
    embed_dim: int
    input_dim: int


ecapa_tdnn_archs = ArchitectureRegistry[EcapaTDNNConfig]("ecapa_tdnn")

ecapa_tdnn_arch = ecapa_tdnn_archs.decorator


@ecapa_tdnn_arch("base")
def _base_ecapa_tdnn() -> EcapaTDNNConfig:
    return EcapaTDNNConfig(
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        groups=[1, 1, 1, 1, 1],
        embed_dim=512,
        input_dim=80,
    )


class EcapaTDNNBuilder:
    """
    Builder module for ECAPA_TDNN model
    """

    config: EcapaTDNNConfig
    device: Optional[Device]
    dtype: Optional[DataType]

    def __init__(
        self,
        config: EcapaTDNNConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param config:
            The configuration to use.
        :param devicev:
            The device on which to initialize modules.
        :param dtype:
            The data type of module parameters and buffers.
        """
        self.config = config

        self.device, self.dtype = device, dtype

    def build_model(self) -> ECAPA_TDNN:
        """Build a model."""
        model = ECAPA_TDNN(
            self.config.channels,
            self.config.kernel_sizes,
            self.config.dilations,
            self.config.attention_channels,
            self.config.res2net_scale,
            self.config.se_channels,
            self.config.global_context,
            self.config.groups,
            self.config.embed_dim,
            self.config.input_dim,
        )
        model.to(device=self.device, dtype=self.dtype)
        return model


def create_ecapa_tdnn_model(
    config: EcapaTDNNConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> ECAPA_TDNN:
    """Create a ECAPA_TDNN model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    return EcapaTDNNBuilder(config, device=device, dtype=dtype).build_model()
