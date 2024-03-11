# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
import importlib
from typing import Any, Dict, Union, List, Optional
from torch import Tensor
from torch.nn import Module
from fairseq2.assets import AssetCard, AssetStore, asset_store
from fairseq2.typing import DataType, Device


@dataclass
class AuxMTLConfig:
    """The auxiliary multi-task learning config that's only used in training"""

    model_type: Module
    """The auxiliary model type (e.g. Linear)"""

    model_config: Dict[str, Any]
    """config to initiate the auxiliary model"""

    index_layer_id: int
    """Which layer representation to attach for auxiliary model"""


class AuxMTLConfigLoader:
    """Loads auxiliary multi-task learning config"""

    def __init__(self, asset_store: AssetStore) -> None:
        """
        :param asset_store:
            The asset store to retrieve the model information.
        """
        self.asset_store = asset_store

    def __call__(self, model_name_or_card: Union[str, AssetCard]) -> AuxMTLConfig:
        """
        :param model_name_or_card:
            The name of the model or an already loaded AssetCard
        """

        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        aux_mtl_dict: Dict = card.field("aux_mtl_config").as_(object)
        try:
            aux_mtl_config = AuxMTLConfig(**aux_mtl_dict)
        except:
            raise KeyError("The auxiliary yaml entry is wrongly named.")
        return aux_mtl_config


load_aux_mtl_config = AuxMTLConfigLoader(asset_store)


class AuxMTLModel(Module):
    """
    Wraps a multi-task learning model w/ index_layer_id
    """
    def __init__(
        self,
        model: Module,
        index_layer_id: int,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.model = model
        self.index_layer_id = index_layer_id
        self.device, self.dtype = device, dtype

    def forward(
        self,
        inner_states: List[Tensor],
    ) -> Tensor:
        model_input = inner_states[self.index_layer_id]
        return self.model(model_input)


class AuxMTLBuilder:
    """Builds a multi-task learning model"""

    def __init__(
        self,
        config: AuxMTLConfig,
        *,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
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

    def build_model(self) -> AuxMTLModel:
        parent_lib, child_cls = self.config.model_type.rsplit(".", 1)
        try:
            parent_module = importlib.import_module(parent_lib)
            model_cls = getattr(parent_module, child_cls)
        except ImportError:
            raise ValueError(
                f"model_type entry ({self.config.model_type}) is wrong or {parent_lib} not installed"
            )

        model: Module = model_cls(**self.config.model_config)
        model = model.to(dtype=self.dtype, device=self.device)
        return AuxMTLModel(
            model=model,
            index_layer_id=self.config.index_layer_id,
        )


def create_aux_mtl_model(
    config: AuxMTLConfig,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> AuxMTLModel:
    """Create a AuxMTLModel.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """

    return AuxMTLBuilder(config, device=device, dtype=dtype).build_model()
