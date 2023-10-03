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
import ggml
from fairseq2.assets import AssetCard
from seamless_communication.models.unity import load_unity_config, load_unity_model

Preprocessor = Callable[[Any], Any]


def convert_model(model_name: str, out: Optional[Path] = None) -> None:
    if out is None:
        out = Path(model_name).with_suffix(".ggml")

    # The type of model depends on the name
    if "unity" in model_name or "seamlessM4T" in model_name:
        model_config = load_unity_config(model_name)
        hparams = flatten_config(dataclasses.asdict(model_config), separator="__")
        print(hparams)
        model = load_unity_model(model_name)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    with out.open("wb") as o:
        write_ggml_file(o, hparams, model.state_dict())

    with out.with_suffix(".hparams.h").open("w") as h:
        h.write(generate_hparams_struct(hparams, "unity_hparams"))


def write_ggml_file(
    out: BufferedWriter, hparams: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
) -> None:
    write_ggml_header(out)

    # Apppend the byte size to the hparams.
    if "model_byte_size" not in hparams:
        # Size of each tensor
        byte_size = sum(x.numel() * x.element_size() for x in state_dict.values())
        # + tensor overhead
        byte_size += ggml.ggml_tensor_overhead() * len(state_dict)
        # + some slack cause I'm bad at math
        byte_size = int(byte_size * 1.2)
        hparams["model_byte_size"] = byte_size
        logging.warning(f"Saving a ggml file with {len(state_dict)} tensors, for an estimated amount of {byte_size / (1024**3)} GGML Gb")
    # 6877961321223123048
    hparams["__end_of_hparams__"] = struct.unpack("l", b"hparams_")[0]

    write_hparams(out, hparams)
    write_state_dict(out, state_dict)


def write_ggml_header(out: BufferedWriter) -> None:
    """Write GGML header (in reverse cause why not)"""
    out.write(b"ggml"[::-1])


def write_hparams(out: BufferedWriter, hparams: Dict[str, Any]) -> None:
    """Write hyper parameters.

    :params hparams:
        flattened dict containing model's hyper parameters.

    """
    # TODO: should we preprend the size of the hparams struct ?
    # this would help catch out of sync writer/loader code
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
    if value.dtype is torch.int64:
        # GGML doesn't ahve int64, downcast it
        value = value.to(dtype=torch.int32)

    if value.ndim == 0:
        # GGML doesn't support scalar as tensors.
        value = value.reshape(1)

    data = value.numpy()
    n_dims = data.ndim
    assert n_dims < 5, "ggml doesn't support 5 dims tensors"
    assert n_dims >= 1, "ggml doesn't support 0 dim tensors"

    ftype = torch_to_ggml_type(value.dtype)
    out.write(struct.pack("i", n_dims))
    out.write(struct.pack("i", ftype))
    for i in range(n_dims):
        # ggml uses long for shape
        out.write(struct.pack("l", data.shape[n_dims - 1 - i]))

    data.tofile(out)

def torch_to_ggml_type(dtype: type) -> int:
    if dtype is torch.float32:
        return ggml.GGML_TYPE_F32
    elif dtype is torch.float16:
        return ggml.GGML_TYPE_F16
    elif dtype is torch.int32:
        return ggml.GGML_TYPE_I32
    else:
        raise NotImplementedError(f"{dtype} is not mapped to a GGML_TYPE")


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


def to_ctype(value: Any) -> Tuple[str, Any]:
    """Transform python type to ctype.

    :params value:
        value to cast into ctype

    :returns:
        A tuple of ctype and cvalue.
    """
    if isinstance(value, int):
        return ("l", value)
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
    if ctype == "l":
        return "std::int64_t"
    if ctype == "f":
        return "float"
    if ctype == "?":
        return "bool"

    raise RuntimeError(
        f"Should not have reached this part." f"Missing cpp translation for {ctype}"
    )


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
    struct = f"struct {struct_name} {{"
    fields = [f"    {get_cpp_type(value)} {key};" for key, value in hparams.items()]
    struct = "\n".join([struct] + fields + ["};\n"])

    valid_fields = [
        key for key, value in hparams.items() if "Error" not in get_cpp_type(value)
    ]
    read_struct = f"void read_{struct_name}({struct_name}& out, std::ifstream &fin) {{"
    read_fields = [
        f"    fin.read((char*) &out.{field}, sizeof(out.{field}));"
        for field in valid_fields
    ]
    read_struct = "\n".join([read_struct] + read_fields + ["};\n"])

    return "\n".join([struct, read_struct])


if __name__ == "__main__":
    import func_argparse

    func_argparse.single_main(convert_model)
