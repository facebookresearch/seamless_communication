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
import math
import torch
import ggml
from typing import Callable
from typing import Optional
from typing import List
from fairseq2.assets import AssetCard
from fairseq2.models.transformer.frontend import TransformerEmbeddingFrontend
from fairseq2.nn import SinusoidalPositionEncoder
from seamless_communication.models.unity import load_unity_config, load_unity_model

Preprocessor = Callable[[Any], Any]

def pos_enc(max_seq_len=4096, encoding_dim=1024):
    weight = torch.empty(
        ((max_seq_len * 2) - 1, encoding_dim), dtype=torch.float32
    )
    # copied from https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/nn/transformer/relative_attention.py#L22
    dtype = torch.float32
    weight = weight.to(dtype)

    positive_w = weight[: max_seq_len]
    negative_w = weight[max_seq_len :]

    device = weight.device

    # (E / 2)
    indices = torch.arange(0, encoding_dim, step=2, device=device, dtype=dtype)

    # (1, E / 2)
    indices = indices.unsqueeze(0)

    # (S)
    steps = torch.arange(max_seq_len, device=device, dtype=dtype)

    # (S, 1)
    steps = steps.unsqueeze(1)

    factors = torch.exp(indices * -math.log(10000) / encoding_dim)

    # (S, 1) x (1, E / 2) -> (S, E / 2)
    factors = torch.matmul(steps, factors)

    flipped_factors = factors.flip([0])

    # A mirrored matrix of sinusoidal positive and negative positional
    # encodings to use in shift trick.
    #
    # [max, ...,  3,  2,  1,  0, -1, -2, -3, ..., min]
    torch.sin(flipped_factors, out=positive_w[:, 0::2])
    torch.cos(flipped_factors, out=positive_w[:, 1::2])

    torch.sin(-1 * factors[1:], out=negative_w[:, 0::2])
    torch.cos(-1 * factors[1:], out=negative_w[:, 1::2])

    return weight

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

    state_dict = model.state_dict()
    fixup_model(model, state_dict)

    with out.open("wb") as o:
        write_ggml_file(o, hparams, state_dict)
        write_layer_config(o, model)

    with out.with_suffix(".hparams.h").open("w") as h:
        h.write(generate_hparams_struct(hparams, "unity_hparams"))


def _nested_getattr(model: Any, name: str) -> Any:
    parts = name.split(".")
    node = model
    for part in parts:
        node = getattr(node, part)
        if node is None:
            return None
    return node


def find_children(model: torch.nn.Module, t: type) -> List[Tuple[str, torch.nn.Module]]:
    queue = list(model._modules.items())
    modules = []
    while queue:
        name, node = queue.pop()
        if node is None:
            continue
        if isinstance(node, t):
            modules.append((name, node))
        for child_name, child_node in node._modules.items():
            queue.append((".".join((name, child_name)), child_node))

    return modules


def fixup_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    # Bake the embedding scaling into the weights
    frontends = find_children(model, TransformerEmbeddingFrontend)
    print("Upgrading the following TransformerEmbeddingFrontend:", [x[0] for x in frontends])
    for name, frontend in frontends:
        embed_weights = state_dict[name + ".embed.weight"]
        state_dict[name + ".embed.weight"] = embed_weights * frontend.scale

    # Sinusoidal embeddings are typically not saved since they are easily recomputed,
    # but this allows to avoid porting the sinusoidal logic to GGML
    pos_encoders = find_children(model, SinusoidalPositionEncoder)
    print("Upgrading the following SinusoidalPositionEncoder:", [x[0] for x in pos_encoders])
    for name, pos_encoder in pos_encoders:
        assert isinstance(pos_encoder.weight, torch.Tensor)
        assert name not in state_dict
        state_dict[name] = pos_encoder.weight

    state_dict["speech_encoder.pos_enc"] = pos_enc()

def write_ggml_file(
    out: BufferedWriter, hparams: Dict[str, Any], state_dict: Dict[str, torch.Tensor]
) -> None:
    write_ggml_header(out)

    # Apppend the byte size to the hparams.
    if "model_byte_size" not in hparams:
        # Size of each tensor
        byte_size = sum(x.numel() * x.element_size() for x in state_dict.values())
        # + tensor overhead
        byte_size += ggml.ggml_tensor_overhead() * (len(state_dict) + 10)
        hparams["model_byte_size"] = byte_size
        logging.warning(
            f"Saving a ggml file with {len(state_dict)} tensors, for an estimated amount of {byte_size / (1024**3)} GGML Gb"
        )
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
        except ValueError:
            logging.warning(f"Skipping config for key {key}={value!r}")
            continue


def write_state_dict(out: BufferedWriter, state_dict: Dict[str, torch.Tensor]) -> None:
    """Write pytorch state dict.

    :paras state_dict:
        state dict returned by pytorch model
    """
    out.write(struct.pack("i", len(state_dict)))
    for key, value in state_dict.items():
        write_string(out, key)
        if key.endswith(".bias") and value.ndim == 1 and "adaptor" not in key:
            # GGML broadcasting isn't as strong as numpy
            value = value.reshape(1, -1)
        if "pointwise_conv" in key: # pointwise_conv / depthwise_conv
            value = value.squeeze(-1)
        if "depthwise_conv" in key:
            value = value.squeeze(1)
        write_tensor(out, value.contiguous())


def write_layer_config(out: BufferedWriter, model: torch.nn.Module) -> None:
    for name, node in find_children(model, torch.nn.Module):
        for k, v in node.__dict__.items():
            # Skip special members. In particular all children module and tensors
            # will be hidden in special dicts `_parameters` and `_modules`
            if k.startswith("_"):
                continue
            # All modules have a "training" flag
            if k == "training":
                continue
            if v is None:
                continue
            try:
                ctype, cvalue = to_ctype(v)
                write_string(out, f"{name}.{k}")
                out.write(struct.pack(ctype, cvalue))
            except ValueError as e:
                logging.warning(f"Skipping config for {name}.{k}={v!r}")
                continue


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
        return ("d", value)
    if isinstance(value, bool):
        return ("l", value)
    if isinstance(value, Enum):
        return ("l", value.value)
    if isinstance(value, tuple) and len(value) == 1:
        return to_ctype(value[0])
    if isinstance(value, str) and len(value) < 8:
        value = bytes(value, "ascii")
        if len(value) < 8:
            value = value + (8 - len(value)) * b"\0"
        return ("l", struct.unpack("l", value)[0])

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
    if ctype == "d":
        return "double"
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
