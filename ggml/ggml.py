"""
We are vendoring https://github.com/abetlen/ggml-python (MIT License)
adding a few utilities to convert between ggml and numpy tensors for testing.
"""

import contextlib
import ctypes
import dataclasses
import functools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, NamedTuple, Tuple, Type, Union

import numpy as np
import torch
import subprocess
import sys

from ctypes_utils import NULLPTR, Ptr, c_fn, c_struct
from third_party_ggml import *

### Helpers


@functools.lru_cache(4)
def numpy_dtype(ggml_type: ctypes.c_int) -> np.dtype:
    if ggml_type == 0:
        # GGML_TYPE_F32  = 0,
        return np.dtype(np.float32)

    if ggml_type == 1:
        # GGML_TYPE_F16  = 1,
        return np.dtype(np.float16)

    if ggml_type == 18:
        return np.dtype(np.int32)

    raise NotImplementedError(f"Can't convert GGML_TYPE({ggml_type}) to a numpy.dtype")


@functools.lru_cache()
def from_numpy_dtype(dtype: np.dtype) -> ctypes.c_int:
    def _ggml_type(name: bytes, value: int) -> ctypes.c_int:
        t = ctypes.c_int(value)
        type_name = ggml_type_name(t)
        if name != type_name:
            raise RuntimeError(
                f"Type {name!r} doesn't have value {value}. ggml.h was probably updated but not ggml.py"
            )
        return t

    if dtype == np.float32:
        return _ggml_type(b"f32", 0)
    elif dtype == np.float16:
        return _ggml_type(b"f16", 1)
    elif dtype == np.dtype("bool"):
        return _ggml_type(b"i8", 16)
    elif dtype == np.int32:
        return _ggml_type(b"i32", 18)

    raise NotImplementedError(f"Can't convert {dtype} to a GGML_TYPE")


def shape(tensor: Union[ggml_tensor, ggml_tensor_p]) -> Tuple[int, ...]:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents
    ndims = tensor.n_dims
    return tuple([tensor.ne[i] for i in range(ndims)[::-1]])


def nb(tensor: Union[ggml_tensor, ggml_tensor_p]) -> Tuple[int, ...]:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents
    return tuple([tensor.nb[i] for i in range(4)])


def ne(tensor: Union[ggml_tensor, ggml_tensor_p]) -> Tuple[int, ...]:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents
    return tuple([tensor.ne[i] for i in range(4)])


def strides(tensor: Union[ggml_tensor, ggml_tensor_p]) -> Tuple[int, ...]:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents
    ndims = tensor.n_dims
    num_bytes = tuple([tensor.nb[i] for i in range(ndims)])
    strides = num_bytes[::-1]
    return strides


def to_numpy(tensor_p: ggml_tensor_p) -> np.ndarray:
    if not ggml_is_contiguous(tensor_p):
        if not _almost_contiguous(tensor_p):
            return _strided_to_numpy(tensor_p)
    tensor = tensor_p.contents

    res = _void_p_to_np_array(tensor.data, shape(tensor), numpy_dtype(tensor.type))

    if ggml_is_transposed(tensor_p):
        # Patch up strides to work with transposed ggml_tensor
        res.strides = strides(tensor)  # type: ignore[assignment]

    return res


def _almost_contiguous(tensor_p: ggml_tensor_p) -> bool:
    """Distinguishes between fully strided and just transposed."""
    tensor = tensor_p.contents
    num_bytes = nb(tensor)
    num_elem = ne(tensor)

    # Sort the axis according to 'num_bytes'
    nbe = sorted(zip(num_bytes, num_elem))
    itemsize = ggml_type_size(tensor.type)
    stride_exp = itemsize
    for stride, e in nbe:
        if stride != stride_exp:
            return False
        stride_exp *= e

    return True


def _strided_to_numpy(tensor_p: ggml_tensor_p) -> np.ndarray:
    if ggml_is_transposed(tensor_p):
        raise NotImplementedError(
            "to_numpy doesn't support tensors both transposed and strided."
        )

    tensor = tensor_p.contents

    n_dim = tensor.n_dims
    t_shape = shape(tensor)
    t_strides = strides(tensor)

    type_size = ggml_type_size(tensor.type)

    full_shape = []
    num_bytes = nb(tensor)

    # Determine the full backing slice of bytes to read.
    # TODO make this work for transposed array
    n = 1
    total_elements = 1
    try:
        for d in range(n_dim - 1):
            n = num_bytes[d + 1] // type_size // n
            full_shape.append(n)
            total_elements *= n
    except ZeroDivisionError:
        logging.warning("Can't convert permuted GGML tensor back to numpy")
        return None
    # We don't need to guess for the first dimension, since this doesn't impact striding.
    full_shape.append(t_shape[0])
    total_elements *= t_shape[0]
    full_shape = full_shape[::-1]

    res = _void_p_to_np_array(tensor.data, tuple(full_shape), numpy_dtype(tensor.type))

    # Extract the correct slice
    res = res.__getitem__(tuple(slice(0, n) for n in t_shape))
    # TODO: we could handle transposition here

    return res


def _void_p_to_np_array(
    data: ctypes.c_void_p, shape: Tuple[int, ...], dtype: np.dtype
) -> np.ndarray:
    # Convert the ggml data pointer to a pointer of bytes
    # This is needed because Python ctypes doesn't have "float16", and `as_array` only works with ctypes
    int_width: type = getattr(ctypes, f"c_uint{8 * dtype.itemsize}")
    ptr = ctypes.cast(data, ctypes.POINTER(int_width))
    # Create a numpy array with the wrong dtype
    int_arr = np.ctypeslib.as_array(ptr, shape=shape)
    # Reinterpret it to the right dtype
    return np.frombuffer(int_arr, dtype=dtype).reshape(shape)


GgmlNElem = ctypes.c_int64 * GGML_MAX_DIMS
GgmlNBytes = ctypes.c_uint64 * GGML_MAX_DIMS


def from_file(
    ctx: ggml_context_p, file: Path, shape: Tuple[int, ...], dtype: type = np.float32
) -> ggml_tensor_p:
    data = np.fromfile(str(file), dtype=dtype).reshape(shape)  # type: ignore
    return from_numpy(ctx, data)


def _shape_to_ne(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    # in GGML ne[0] indicates the contiguous dimension, ie the last one in numpy and torch
    ne = shape[::-1]
    if len(ne) >= GGML_MAX_DIMS:
        return ne  # type: ignore

    # ne is always of the same length
    padding = (1,) * (GGML_MAX_DIMS - len(ne))
    return ne + padding  # type: ignore


def _compute_nbytes(
    ne: Tuple[int, int, int, int], type: ctypes.c_int
) -> Tuple[int, int, int, int]:
    nb0 = ggml_type_size(type)
    nb1 = nb0 * (ne[0] // ggml_blck_size(type))
    nb2 = nb1 * ne[1]
    nb3 = nb2 * ne[2]
    return (nb0, nb1, nb2, nb3)


def from_numpy(
    ctx: ggml_context_p, array: Union[np.ndarray, "torch.Tensor"], name: bytes = b""
) -> Ptr[ggml_tensor]:
    if type(array).__name__ == "Tensor":
        array = array.numpy()
    # Create an empty tensor so we don't allocate memory for the data pointer
    gtype = from_numpy_dtype(array.dtype)
    tensor_p = ggml_new_tensor_1d(ctx, gtype, 0)
    # Fill out the correct dimensions and shape.
    tensor_p.contents.n_dims = array.ndim
    ne = _shape_to_ne(array.shape)
    tensor_p.contents.ne = GgmlNElem(*ne)
    tensor_p.contents.nb = GgmlNBytes(*_compute_nbytes(ne, gtype))
    # point the tensor data to the content of the numpy array.
    tensor_p.contents.data = array.ctypes.data_as(ctypes.c_void_p)
    # print(f"array: {array.shape} @0x{array.ctypes.data_as(ctypes.c_void_p)}")
    # print(f"tensor_p: {shape(tensor_p)} @0x{tensor_p.contents.data:x}")

    # prevent the underlying numpy array to be freed
    setattr(tensor_p, "__data", array)
    if name:
        ggml_set_name(tensor_p, name)
    return tensor_p  # type: ignore


def ggml_can_mul_mat(t0: ggml_tensor_p, t1: ggml_tensor_p) -> bool:
    assert GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function"

    return (
        (t0.contents.ne[0] == t1.contents.ne[0])
        and (t1.contents.ne[2] % t0.contents.ne[2] == 0)
        and (t1.contents.ne[3] % t0.contents.ne[3] == 0)
    )


def nodes(gf: ggml_cgraph) -> Dict[bytes, ggml_tensor_p]:
    res = {}
    for i in range(gf.n_nodes):
        name = gf.nodes[i].contents.name
        res[name] = gf.nodes[i]
    return res


def leafs(gf: ggml_cgraph) -> Dict[bytes, ggml_tensor_p]:
    res = {}
    for i in range(gf.n_leafs):
        name = gf.leafs[i].contents.name
        res[name] = gf.leafs[i]
    return res


class NativeObj:
    AllocFn = Callable[[], ctypes.c_void_p]
    FreeFn = Callable[[ctypes.c_void_p], None]
    _cache: Dict[str, Tuple[AllocFn, FreeFn]] = {}

    @classmethod
    def _init_c_func(cls, kind: str) -> Tuple[AllocFn, FreeFn]:
        if kind in cls._cache:
            return cls._cache[kind]

        alloc_fn = getattr(lib, f"{kind}_alloc")
        alloc_fn.argtypes = []
        alloc_fn.restype = ctypes.c_void_p

        free_fn = getattr(lib, f"{kind}_free")
        free_fn.argtypes = [ctypes.c_void_p]
        free_fn.restype = None

        cls._cache[kind] = (alloc_fn, free_fn)
        return (alloc_fn, free_fn)

    def __init__(self, kind: str, ptr: ctypes.c_void_p = NULL):
        self.kind = kind
        alloc_fn, self._free_fn = self._init_c_func(kind)
        self.ptr = alloc_fn() if ptr is None else ptr
        # print(self)

    def free(self) -> None:
        if self.ptr is not None:
            self._free_fn(self.ptr)
            # print(f"freeing {self}")
            self.ptr = NULL

    def __enter__(self) -> ctypes.c_void_p:
        return self.ptr

    def __exit__(self, *args: Any) -> None:
        self.free()

    def __del__(self) -> None:
        self.free()

    def __repr__(self) -> str:
        return f"<{self.kind} native object at 0x{self.ptr:x}>"


def MeasureArena() -> NativeObj:
    return NativeObj("ggml_allocr", ggml_allocr_new_measure(GGML_MEM_ALIGN))


def FixedSizeArena(mem_size: int) -> NativeObj:
    memory = torch.zeros(mem_size, dtype=torch.uint8)
    allocr = ggml_allocr_new(
        ctypes.c_void_p(memory.data_ptr()), mem_size, GGML_MEM_ALIGN
    )
    arena = NativeObj("ggml_allocr", allocr)
    # Add a reference from the arena object to the underlying tensor, otherwise it will be freed to early.
    setattr(arena, "__memory", memory)
    return arena


lib.fairseq2_model_set_inference_ctx.argtypes = [ctypes.c_void_p, ggml_context_p]


def Fairseq2Model() -> NativeObj:
    return NativeObj("fairseq2_model")


lib.std_string_alloc.argtypes = [ctypes.c_char_p]
lib.std_string_alloc.restype = ctypes.c_void_p
lib.std_string_free.argtypes = [ctypes.c_void_p]
lib.std_string_free.restype = None
NativeObj._cache["std_string"] = (lib.std_string_alloc, lib.std_string_free)


def CppStr(content: str) -> NativeObj:
    c_str = ctypes.create_string_buffer(content.encode("utf-8"))
    cpp_str = lib.std_string_alloc(c_str)
    return NativeObj("std_string", cpp_str)


lib.load_fairseq2_ggml_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.load_fairseq2_ggml_file.restype = ctypes.c_int


def load_fairseq2_ggml_file(model_file: Path) -> NativeObj:
    model = Fairseq2Model()
    bytes_file = ctypes.create_string_buffer(str(model_file).encode("utf-8"))
    err = lib.load_fairseq2_ggml_file(model.ptr, bytes_file)
    if err:
        raise Exception("Failed to load model")
    return model


# lib.unity_audio_encoder_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
# lib.unity_audio_encoder_graph.restype = ctypes.POINTER(ggml_cgraph)


# def unity_audio_encoder_graph(model: NativeObj, tensor: ggml_tensor_p) -> ggml_cgraph_p:
#     return lib.unity_audio_encoder_graph(model.ptr, tensor)  # type: ignore


# lib.unity_eval.argtypes = [
#     ctypes.c_void_p,
#     ctypes.c_void_p,
#     ctypes.POINTER(ggml_tensor),
#     ctypes.c_int,
# ]
# lib.unity_eval.restype = ctypes.POINTER(ggml_cgraph)


# def unity_eval(
#     allocr: ctypes.c_void_p, model: NativeObj, tensor: ggml_tensor_p, n_threads: int
# ) -> ggml_cgraph_p:
#     return lib.unity_eval(allocr, model.ptr, tensor, n_threads)


_FORWARD_CACHE: Dict[str, Callable[..., ggml_tensor_p]] = {}


def forward(
    layer_name: str, model: ctypes.c_void_p, prefix: str, *inputs: ggml_tensor_p
) -> ggml_tensor_p:
    fwd: Any = _FORWARD_CACHE.get(layer_name)
    if fwd is None:
        fwd = getattr(lib, layer_name + "_forward")
        num_inputs = len(inputs)
        fwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [
            ctypes.POINTER(ggml_tensor)
        ] * num_inputs
        fwd.restype = ctypes.POINTER(ggml_tensor)
        _FORWARD_CACHE[layer_name] = fwd

    with CppStr(prefix) as std_prefix:
        return fwd(model, std_prefix, *inputs)  # ignore: type[no-any-return]


def build_and_compute(
    ctx: ggml_context_p, tensor: ggml_tensor_p, num_threads: int = 1, dump: Union[bool, str] = False
) -> ggml_cgraph:
    gf = ggml_build_forward(tensor)
    need_alloc = tensor.contents.data == NULLPTR
    if need_alloc:
        alloc = FixedSizeArena(1024 * 1024 * 1024 * 2)
        ggml_allocr_alloc_graph(alloc.ptr, ctypes.pointer(gf))
        setattr(tensor, "__data", alloc)
    if dump:
        if dump == True:
            dump = f"dot/{sys._getframe(1).f_code.co_name}"
        ggml_graph_dump_dot(ctypes.pointer(gf), NULLPTR, dump.encode("ascii"))
        # subprocess.run(["dot", "-Tsvg", "-O", dump])
    ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), num_threads)
    return gf


@c_fn(lib)
def causal_attention_mask(
    ctx: ggml_context_p, seqs: Ptr[ggml_tensor]
) -> Ptr[ggml_tensor]:
    ...


@c_fn(lib)
def ggml_slice(
    ctx: ggml_context_p,
    a: Ptr[ggml_tensor],
    axis: int,
    start: ctypes.c_int64,
    end: ctypes.c_int64,
) -> Ptr[ggml_tensor]:
    ...


@c_fn(lib)
def ggml_flatten_1d(
    ctx: ggml_context_p, a: Ptr[ggml_tensor], dim: int
) -> Ptr[ggml_tensor]:
    return a


@c_fn(lib)
def ggml_unflatten_1d(
    ctx: ggml_context_p, a: Ptr[ggml_tensor], dim: int, num_el: int
) -> Ptr[ggml_tensor]:
    return a


@c_struct
@dataclasses.dataclass
class SequenceGeneratorOptions:
    beam_size: int
    min_seq_len: int = 5
    soft_max_seq_len_a: float = 1.0
    soft_max_seq_len_b: int = 200
    hard_max_seq_len: int = 1024
    len_penalty: float = 1.0
    unk_penalty: float = 0.0
    normalize_scores: bool = True


@c_struct
@dataclasses.dataclass
class SequenceGeneratorJob:
    opts: SequenceGeneratorOptions
    prefix_seq: Ptr[ggml_tensor]
    pad_idx: int
    unk_idx: int
    bos_idx: int
    eos_idx: int
    num_threads: int = 1


@c_struct
class Hypothesis:
    seq: Ptr[ggml_tensor]
    """The generated sequence."""

    score: float
    """The score of the hypothesis."""

    step_scores: Ptr[ggml_tensor]
    """The score of each individual sequence step."""


@c_fn(lib)
def generate_sequence(
    model: ctypes.c_void_p,
    job: Ptr[SequenceGeneratorJob],
    encoder_output: Ptr[ggml_tensor],
    encoder_padding_mask: Ptr[ggml_tensor],
    result_ctx: ggml_context_p,
) -> Ptr[Hypothesis]:
    ...


@c_fn(lib)
def _testing_return_hypothesis_ptr(ctx: ggml_context_p) -> Ptr[Hypothesis]:
    return Ptr()


@c_fn(lib)
def fairseq2_model_layer_config_int(model: ctypes.c_void_p, name: bytes) -> int:
    return -1


@c_fn(lib.fairseq2_kv_cache_alloc)
def _fairseq2_kv_cache_alloc(
    model: ctypes.c_void_p, ctx: ctypes.c_void_p, beam_size: int, max_seq_len: int
) -> None:
    pass


@c_fn(lib.fairseq2_kv_cache_reset)
def _fairseq2_kv_cache_reset(model: ctypes.c_void_p) -> None:
    pass


@contextlib.contextmanager
def fairseq2_kv_cache_alloc(
    model: ctypes.c_void_p, kv_cache_size: int, beam_size: int, max_seq_len: int
) -> Iterator[None]:

    memory = torch.zeros(kv_cache_size, dtype=torch.uint8)
    ctx = ggml_init(
        params=ggml_init_params(
            mem_size=kv_cache_size,
            mem_buffer=ctypes.c_void_p(memory.data_ptr()),
            no_alloc=False,
        )
    )
    _fairseq2_kv_cache_alloc(model, ctx, beam_size, max_seq_len)
    try:
        yield
    finally:
        _fairseq2_kv_cache_reset(model)
        ggml_free(ctx)


@c_fn(lib)
def fairseq2_spm_tokenize(
    model: ctypes.c_void_p, text: bytes, out: Ptr[ggml_tensor]
) -> None:
    pass


@c_fn(lib)
def fairseq2_spm_detokenize(
    model: ctypes.c_void_p, tensor: Ptr[ggml_tensor], out: ctypes.Array[ctypes.c_char]
) -> ctypes.c_size_t:
    return 0
