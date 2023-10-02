# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import struct
from enum import Enum
from io import BufferedWriter
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from fairseq2.assets import AssetCard
from seamless_communication.models.unity import load_unity_config, load_unity_model

Preprocessor = Callable[[Any], Any]


def to_ctype(value: Any) -> Tuple[str, Any]:
    """Transform python type to ctype.

    :params value:
        value to cast into ctype

    :returns:
        A tuple of ctype and cvalue.
    """
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, float):
        return ("f", value)
    if isinstance(value, bool):
        return ("?", value)
    if isinstance(value, Enum):
        return ("i", value.value)

    raise ValueError(f"Unsupported type {type(value)}")


def get_cpp_type(value: Any) -> str:
    """Return equivalent cpp type in string format

    :params value:
        value to cast into ctype

    :returns:
        str containing cpp type
    """
    # used to have compatibility between types
    try:
        ctype, _ = to_ctype(value)
    except ValueError as e:
        return f"// Error: {e}"

    if ctype == "i":
        return "std::int32_t"
    if ctype == "f":
        return "std::float32"
    if ctype == "?":
        return "bool"

    raise RuntimeError(
        f"Should not have reached this part." f"Missing cpp translation for {ctype}"
    )


def write_ggml_header(out: BufferedWriter) -> None:
    """Write GGML header"""
    out.write(b"ggml")


def write_hparams(out: BufferedWriter, hparams: Dict[str, Any]) -> None:
    """Write hyper parameters.

    :params hparams:
        flattened dict containing model's hyper parameters.

    """
    for key, value in hparams.items():
        try:
            # TODO: this is not cross platform, what's the standard way of writing hparams in GGML ?
            ctype, cvalue = to_ctype(value)
            out.write(struct.pack(ctype, cvalue))
        except ValueError as e:
            logging.warning(f"[Warning] {e}. Skipping config for key {key}")
            continue


def write_state_dict(out: BufferedWriter, state_dict: Dict[str, torch.Tensor]) -> None:
    """Write pytorch state dict.

    :paras state_dict:
        state dict returned by pytorch model
    """
    for key, value in state_dict.items():
        write_string(out, key)
        write_tensor(out, value)


def write_string(out: BufferedWriter, value: str) -> None:
    """Write string in utf-8 format.

    :params value:
        string value to dump.
    """
    str_ = value.encode("utf-8")
    out.write(struct.pack("i", len(str_)))
    out.write(str_)


def write_tensor(out: BufferedWriter, value: torch.Tensor) -> None:
    """Write torch tensor in ggml format.

    First we save the number of dimensions and the dtype.
    Then we save the data as numpy array.

    :params value:
        Tensor to dump.
    """
    data = value.squeeze().numpy()
    n_dims = len(data.shape)

    # TODO: Convert to fp16 when necessary!
    ftype = 0

    out.write(struct.pack("ii", n_dims, ftype))
    for i in range(n_dims):
        out.write(struct.pack("i", data.shape[n_dims - 1 - i]))

    data.tofile(out)


def write_ggml_file(
    out: BufferedWriter, hparams: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
) -> None:
    write_ggml_header(out)
    write_hparams(out, hparams)
    write_state_dict(out, state_dict)


def flatten_config(
    config: Dict[str, Any],
    separator: str,
    config_preprocessor: Optional[Preprocessor] = None,
) -> Dict[str, Any]:
    """Flatten nested dictionnary

    :param config:
        nested dictionnary containing model config.
    :param separator:
            string separator used when flattening nested hparams
    :param config_preprocessor:
        Preprocessor used for config/hparams values

    :returns:
        flat dictionnary
    """

    if config_preprocessor is None:
        config_preprocessor = lambda x: x

    def __flatten(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        result = {}
        for key in config:
            new_key = f"{prefix}{key}"
            if isinstance(config[key], dict):
                nested_result = __flatten(config[key], f"{new_key}{separator}")
                result.update(nested_result)
            else:
                new_config = config_preprocessor(config[key])
                if new_config is not None:
                    result[new_key] = config[key]

        return result

    return __flatten(config)


def generate_hparams_struct(
    hparams: Dict[str, Any],
    struct_name: str,
) -> str:
    """Generate a c++ struct to hold the model hyper-parameters.

    :param hparams:
        Flattened config of the model.
    :param struct_name:
        Name of the generated struct.
    """
    struct = f"struct {struct_name} {{\n"
    fields = "\n".join(
        [f"    {get_cpp_type(value)} {key};" for key, value in hparams.items()]
    )

    return struct + fields + "\n};\n"


def main(model_name: str, out: Optional[Path] = None) -> None:
    if out is None:
        out = Path(model_name).with_suffix(".ggml")

    # The type of model depends on the name
    if "unity" in model_name or "seamlessM4T" in model_name:
        model_config = load_unity_config(model_name)
        hparams = flatten_config(dataclasses.asdict(model_config), separator="__")
        model = load_unity_model(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    with out.open("wb") as o:
        write_ggml_file(o, hparams, model.state_dict())

    with out.with_suffix(".hparams.h").open("w") as h:
        h.write(generate_hparams_struct(hparams, model_name + "_hparams"))


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(main)
