# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import dataclasses
import logging
import math
import struct
from enum import Enum
from io import BufferedWriter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from fairseq2.assets import AssetCard
from fairseq2.models.transformer.frontend import TransformerEmbeddingFrontend
from fairseq2.nn import SinusoidalPositionEncoder
from fairseq2.nn.transformer import RelativePositionalEncoding
from seamless_communication.models import unity

import ggml
import re

Preprocessor = Callable[[Any], Any]
log = logging.getLogger("ggml_convert")


def convert_model(
    model_name: Union[str, torch.nn.Module],
    out: Optional[Path] = None,
    layers: str = "",
    hparams: Optional[Dict[str, Any]] = None,
    vocab: Optional[List[Tuple[str, float]]] = None,
    fp16: bool = False,
) -> None:
    if isinstance(model_name, str):
        # Load the corresponding fairseq2 model
        if out is None:
            out = Path(model_name).with_suffix(".ggml")

        # The type of model depends on the name
        if "unity" in model_name or "seamlessM4T" in model_name:
            if hparams is None:
                model_config = unity.load_unity_config(model_name)
                hparams = flatten_config(
                    dataclasses.asdict(model_config), separator="__"
                )
                log.info(hparams)
            model = unity.load_unity_model(model_name)
            if vocab is None:
                tokenizer = unity.load_unity_text_tokenizer(model_name)
                vocab = read_vocab(tokenizer)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    else:
        # Use the model passed explicitly
        assert (
            out is not None
        ), "output path is required when explicitly passing a module"
        hparams = hparams or {}
        model = model_name

    state_dict = model.state_dict()
    if layers:
        state_dict = {k: v for k, v in state_dict.items() if re.match(layers, k)}
    fixup_model(model, state_dict, layer_filter=layers)
    layer_config = read_layer_config(model, layer_filter=layers)
    vocab = vocab or []
    write_ggml_file(out, hparams, layer_config, vocab, state_dict, fp16)


def _nested_getattr(model: Any, name: str) -> Any:
    parts = name.split(".")
    node = model
    for part in parts:
        node = getattr(node, part)
        if node is None:
            return None
    return node


def find_children(model: torch.nn.Module, t: type, layer_filter: str = "") -> List[Tuple[str, torch.nn.Module]]:
    queue = list(model._modules.items())
    modules = []
    while queue:
        name, node = queue.pop()
        if node is None:
            continue
        if layer_filter and not re.match(layer_filter, name):
            continue
        if isinstance(node, t):
            modules.append((name, node))
        for child_name, child_node in node._modules.items():
            queue.append((".".join((name, child_name)), child_node))

    return modules


def fixup_model(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], layer_filter: str) -> None:
    # Bake the embedding scaling into the weights
    frontends = find_children(model, TransformerEmbeddingFrontend, layer_filter)
    if frontends:
        log.info(
            "Upgrading the following TransformerEmbeddingFrontend: {}",
            [x[0] for x in frontends],
        )
    for name, frontend in frontends:
        embed_weights = state_dict[name + ".embed.weight"]
        state_dict[name + ".embed.weight"] = embed_weights * frontend.scale

    # Sinusoidal embeddings are typically not saved since they are easily recomputed,
    # but this allows to avoid porting the sinusoidal logic to GGML
    pos_encoders = find_children(model, SinusoidalPositionEncoder, layer_filter)
    if pos_encoders:
        log.info(
            "Upgrading the following SinusoidalPositionEncoder: {}",
            [x[0] for x in pos_encoders],
        )
    for name, pos_encoder in pos_encoders:
        assert isinstance(pos_encoder.freqs, torch.Tensor)
        assert name not in state_dict
        state_dict[name] = pos_encoder.freqs

    relative_pos_encs = find_children(model, RelativePositionalEncoding, layer_filter)
    # speech_encoder has several copies of the relative_pos_enc module.
    # For efficiency reasons we only make one copy of it to GGML.
    if relative_pos_encs:
        log.info("Merging all speech_encoder RelativePositionalEncoding into one.")
        _, rel_pos_enc = relative_pos_encs[0]
        assert isinstance(rel_pos_enc.freqs, torch.Tensor)
        state_dict["speech_encoder.pos_enc"] = rel_pos_enc.freqs


def convert_to_fp16(state_dict: Dict[str, torch.Tensor]) -> None:
    for k in state_dict:
        v = state_dict[k]
        if v.dtype != torch.float32:
            # ignore int tensors
            continue
        state_dict[k] = v.to(torch.float16)


def read_vocab(tokenizer: Any) -> List[Tuple[str, float]]:
    vocab_info = tokenizer.vocab_info
    vocab = [
        (tokenizer.model.index_to_token(i).replace("▁", " "), -i)
        for i in range(vocab_info.size)
    ]
    return vocab  # type: ignore[return-value]


def write_ggml_file(
    out: Path,
    hparams: Dict[str, Any],
    layer_config: Dict[str, Any],
    vocab: List[Tuple[str, float]],
    state_dict: Dict[str, torch.Tensor],
    fp16: bool,
) -> None:
    with out.open("wb") as o:
        write_ggml_header(o)
        write_hparams(o, hparams)
        write_hparams(o, layer_config)
        write_vocab(o, vocab)
        write_state_dict(o, state_dict, fp16)


def write_ggml_header(out: BufferedWriter) -> None:
    """Write GGML header (in reverse cause big-endian)"""
    out.write(b"ggml"[::-1])


def write_hparams(out: BufferedWriter, hparams: Dict[str, Any]) -> None:
    """Write hyper parameters.

    :params hparams:
        flattened dict containing model's hyper parameters.

    """
    simple_vals = {}
    for key, value in hparams.items():
        try:
            simple_vals[key] = to_ctype(value)
        except ValueError:
            logging.warning(f"Skipping config for key {key}={value!r}")
            continue

    out.write(struct.pack("<q", len(simple_vals)))
    for key, (ctype, cvalue) in simple_vals.items():
        write_string(out, key)
        b = struct.pack(ctype, cvalue)
        assert len(b) == 8
        out.write(b)

    logging.info(f"Saved {len(simple_vals)} params.")


def write_vocab(out: BufferedWriter, vocab: List[Tuple[str, float]]) -> None:
    out.write(struct.pack("<q", len(vocab)))

    # Write all words concatenated in a buffer
    words = [bytes(w, "utf8") for w, score in vocab]
    packed_words = b"\0".join(words)
    # We use i32 to allow reusing the string loading codes
    packed_len = struct.pack("<i", len(packed_words))
    out.write(packed_len)
    out.write(packed_words)

    lengths = torch.tensor([len(w) for w in words], dtype=torch.int8)
    write_tensor(out, lengths)

    scores = torch.tensor([score for w, score in vocab], dtype=torch.float32)
    write_tensor(out, scores)


def write_state_dict(
    out: BufferedWriter, state_dict: Dict[str, torch.Tensor], fp16: bool
) -> None:
    """Write pytorch state dict.

    :params state_dict:
        state dict returned by pytorch model
    :params fp16:
        convert float32 tensors to float16 on disk
    """
    out.write(struct.pack("<q", len(state_dict)))
    # True size of each tensor (before downcasting to float16)
    true_byte_size = sum(x.numel() * x.element_size() for x in state_dict.values())
    out.write(struct.pack("<q", true_byte_size))

    GB = 1024**3
    if not fp16:
        log.warning(
            f"Saving a ggml file with {len(state_dict)} tensors, totalling {true_byte_size / GB:.3f}Gb"
        )
    else:

        def _fp16_byte_size(x: torch.Tensor) -> int:
            full_byte_size = x.numel() * x.element_size()
            if fp16 and x.dtype == torch.float32:
                full_byte_size //= 2
            return full_byte_size

        # Compressed size
        compressed_byte_size = sum(_fp16_byte_size(x) for x in state_dict.values())
        log.warning(
            f"Saving a ggml file with {len(state_dict)} tensors, totalling {true_byte_size / GB:.3f}Gb compressed to {compressed_byte_size / GB:.3f}"
        )

    for key, value in state_dict.items():
        write_string(out, key)
        if key.endswith(".bias") and value.ndim == 1 and "adaptor" not in key:
            # GGML broadcasting isn't as strong as numpy
            value = value.reshape(1, -1)
        if "pointwise_conv" in key:  # pointwise_conv / depthwise_conv
            value = value.squeeze(-1)
        if "depthwise_conv" in key:
            value = value.squeeze(1)
        if fp16 and value.dtype == torch.float32:
            value = value.to(torch.float16)
        write_tensor(out, value.contiguous())


def write_string(out: BufferedWriter, value: str) -> None:
    """Write string in utf-8 format.

    :params value:
        string value to dump.
    """
    str_ = value.encode("utf-8")
    packed_len = struct.pack("<i", len(str_))
    assert len(packed_len) == 4
    out.write(packed_len)
    out.write(str_)


def write_tensor(out: BufferedWriter, value: torch.Tensor) -> None:
    """Write torch tensor in ggml format.

    First we save the number of dimensions and the dtype.
    Then we save the data as numpy array.

    :params value:
        Tensor to dump.
    """
    if value.dtype is torch.int64:
        # GGML doesn't have int64, downcast it
        value = value.to(dtype=torch.int32)

    if value.ndim == 0:
        # GGML doesn't support scalar as tensors.
        value = value.reshape(1)

    data = value.numpy()
    n_dims = data.ndim
    assert n_dims < 5, "ggml doesn't support 5 dims tensors"
    assert n_dims >= 1, "ggml doesn't support 0 dim tensors"

    ftype = torch_to_ggml_type(value.dtype)
    out.write(struct.pack("<i", n_dims))
    out.write(struct.pack("<i", ftype))
    for i in range(n_dims):
        # ggml uses long for shape
        out.write(struct.pack("<q", data.shape[n_dims - 1 - i]))

    data.tofile(out)


def torch_to_ggml_type(dtype: torch.dtype) -> int:
    if dtype is torch.float32:
        return ggml.GGML_TYPE_F32
    elif dtype is torch.float16:
        return ggml.GGML_TYPE_F16
    elif dtype is torch.int32:
        return ggml.GGML_TYPE_I32
    elif dtype is torch.int8:
        return ggml.GGML_TYPE_I8
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


def read_layer_config(model: torch.nn.Module, layer_filter: str) -> Dict[str, Any]:
    layer_config = {}

    def _append_node_config(node: Any, prefix: str) -> None:
        for k, v in node.__dict__.items():
            # Skip special members. In particular all children module and tensors
            # will be hidden in special dicts `_parameters` and `_modules`
            if k.startswith("_"):
                continue
            # All modules have a "training" flag
            if k in ("training", "init_fn"):
                continue
            if v is None:
                continue

            try:
                to_ctype(v)
            except ValueError:
                log.warning(f"Skipping layer config {k}={v!r}")
                continue
            layer_config[prefix + k] = v

    _append_node_config(model, "")
    for name, node in find_children(model, torch.nn.Module, layer_filter):
        _append_node_config(node, name + ".")
    return layer_config


def to_ctype(value: Any) -> Tuple[str, Any]:
    """Transform python type to ctype.

    Note: we always use little-endian and 8-byte types.
    This make the format independent of the current platform.

    :params value:
        value to cast into ctype

    :returns:
        A tuple of ctype and cvalue.
    """
    if isinstance(value, int):
        return ("<q", value)
    if isinstance(value, float):
        return ("<d", value)
    if isinstance(value, bool):
        return ("<q", value)
    if isinstance(value, Enum):
        return ("<q", value.value)
    if isinstance(value, tuple) and len(value) == 1:
        return to_ctype(value[0])
    if isinstance(value, str) and len(value) < 8:
        value = bytes(value, "ascii")
        if len(value) < 8:
            value = value + (8 - len(value)) * b"\0"
        return ("8s", value)

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
