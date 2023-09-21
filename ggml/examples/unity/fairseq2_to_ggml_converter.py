# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from pathlib import Path
from typing import Any, Callable, Optional, Union

from fairseq2.assets import AssetCard

from ggml.examples.unity.buffered_ggml_writer import BufferedGGMLWriter
from ggml.examples.unity.type_utils import get_cpp_type
from seamless_communication.models.unity import (
    load_unity_config,
    load_unity_model
)

Preprocessor = Callable[[Any], Any]


class Fairseq2ToGGMLConverter:
    """Converter from fairseq2 format to GGML format"""

    config_preprocessor: Preprocessor
    nested_params_separtor: str

    def __init__(
        self,
        nested_params_separtor: str = ".",
        config_preprocessor: Optional[Preprocessor] = None,
    ) -> None:
        """
        :param nested_params_separtor:
            string separator used when flattening nested hparams
        :param config_preprocessor:
            Preprocessor used for config/hparams values
        """
        self.config_preprocessor = config_preprocessor or (lambda v: v)
        self.nested_params_separtor = nested_params_separtor

    def convert_to_ggml(
        self,
        model_name_or_card: Union[str, AssetCard],
        output_file: Path
    ) -> None:
        """Load model from card, convert to ggml format and save result.

        :param model_name_or_card:
            The name or asset card of the model to load.
        :param output_file:
            File path to store binary output.
        """
        hparams = self._load_config(model_name_or_card)
        state_dict = self._load_state_dict(model_name_or_card)

        buffer = output_file.open("wb")

        ggml_writer = BufferedGGMLWriter(buffer)

        ggml_writer.write_magic_hex()
        ggml_writer.write_hparams(hparams)
        ggml_writer.write_state_dict(state_dict)

        buffer.close()

    def generate_hparams_struct(
        self,
        model_name_or_card: Union[str, AssetCard],
        struct_name: str,
    ) -> str:
        """Transform config to c++ struct

        :param model_name_or_card:
            The name or asset card of the model to load.
        :param output_file:
            File path to store binary output.
        """
        hparams = self._load_config(model_name_or_card)
        result = f"struct {struct_name} {{\n"
        for key, value in hparams.items():
            result = f"{result}\t{get_cpp_type(value)} {key};\n"

        result = f"{result}}};"

        return result

    def _load_config(
        self,
        model_name_or_card: Union[str, AssetCard]
    ) -> dict:
        """Load model config and transform it to flattened dict.

        :param model_name_or_card:
            The name or asset card of the model to load.

        :returns:
            Flat dictionnary containing all hyper parameters.
        """
        model_config = load_unity_config(model_name_or_card)
        model_config_dict = dataclasses.asdict(model_config)
        flattened = self.__flatten(model_config_dict)

        return flattened

    def _load_state_dict(
        self,
        model_name_or_card: Union[str, AssetCard]
    ) -> dict:
        """Load model and return state dict.

        :param model_name_or_card:
            The name or asset card of the model to load.

        :returns:
            State dict returned by pytorch model.
        """
        model = load_unity_model(model_name_or_card)

        return model.state_dict()

    def __flatten(
        self,
        config: dict
    ) -> dict:
        """Flatten nested dictionnary

        :param config:
            nested dictionnary containing model config.

        :returns:
            flat dictionnary
        """
        return self.__flatten_recursive(config, '')

    def __flatten_recursive(
        self,
        config: dict,
        prefix: str
    ) -> dict:
        """Recursive method used to flatten nested dictionnary"""
        result = {}
        for key in config:
            new_key = f"{prefix}{key}"
            if isinstance(config[key], dict):
                nested_result = self.__flatten_recursive(
                    config[key],
                    f"{new_key}{self.nested_params_separtor}"
                )
                result.update(nested_result)
            else:
                new_config = self.config_preprocessor(config[key])
                if new_config is not None:
                    result[new_key] = config[key]

        return result
