"""
We are vendoring https://github.com/abetlen/ggml-python (MIT License)
adding a few utilities to convert between ggml and numpy tensors for testing.
"""

import numpy as np
import ctypes
import torch
import functools
from pathlib import Path
from typing import Self
from typing import Dict
from typing import Callable
from typing import Any
from typing import Tuple
from typing import Union
from typing import Type

from third_party_ggml import *

### Helpers


def numpy_dtype(ggml_type: ctypes.c_int) -> type:
    if ggml_type == 0:
        # GGML_TYPE_F32  = 0,
        return np.float32

    if ggml_type == 1:
        # GGML_TYPE_F16  = 1,
        return np.float16

    raise NotImplementedError(f"Can't convert GGML_TYPE({ggml_type}) to a numpy.dtype")


def from_numpy_dtype(dtype: np.dtype) -> ctypes.c_int:
    if dtype == np.float32:
        return ctypes.c_int(0)
    elif dtype == np.float16:
        return ctypes.c_int(1)
    raise NotImplementedError(f"Can't convert {dtype} to a GGML_TYPE")


def shape(tensor: Union[ggml_tensor, ggml_tensor_p]) -> Tuple[int, ...]:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents
    ndims = tensor.n_dims
    return tuple([tensor.ne[i] for i in range(ndims)])


def strides(tensor: Union[ggml_tensor, ggml_tensor_p]) -> Tuple[int, ...]:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents
    ndims = tensor.n_dims
    return tuple([tensor.nb[i] for i in range(ndims)])


def to_numpy(tensor: Union[ggml_tensor, ggml_tensor_p]) -> np.ndarray:
    if isinstance(tensor, ctypes._Pointer):
        tensor = tensor.contents

    t_shape = shape(tensor)

    # Convert the ggml data pointer to a pointer to ints with the same size (float16 -> uint16)
    # This is needed because Python ctypes doesn't have "float16", and as_array only works with ctypes pointer
    type_size = ggml_type_size(tensor.type)
    int_width: type = getattr(ctypes, f"c_uint{8 * type_size}")
    ptr = ctypes.cast(tensor.data, ctypes.POINTER(int_width))
    # Create a numpy array with the wrong dtype
    int_arr = np.ctypeslib.as_array(ptr, shape=t_shape)
    # Reinterpret it to the right dtype
    res = np.frombuffer(int_arr, dtype=numpy_dtype(tensor.type)).reshape(t_shape)

    # TODO: assert strides / check contiguous
    # assert strides(tensor) == res.strides, "TODO: support strided tensor"
    return res


GgmlShape = ctypes.c_int64 * GGML_MAX_DIMS


def from_file(
    ctx: ggml_context_p, file: Path, shape: Tuple[int, ...], dtype: type = np.float32
) -> ggml_tensor_p:
    data = np.fromfile(str(file), dtype=dtype).reshape(shape)  # type: ignore
    return from_numpy(ctx, data)


def _pad_shape(shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    if len(shape) >= 4:
        return shape

    padding = (1,) * (4 - len(shape))
    return shape + padding  # type: ignore


def from_numpy(ctx: ggml_context_p, array: Union[np.ndarray, "torch.Tensor"]) -> ggml_tensor_p:
    if type(array).__name__ == "Tensor":
        array = array.numpy()
    tensor_p = ggml_new_tensor(ctx, from_numpy_dtype(array.dtype), 1, GgmlShape())
    tensor_p.contents.n_dims = array.ndim
    tensor_p.contents.data = array.ctypes.data_as(ctypes.c_void_p)
    tensor_p.contents.ne = GgmlShape(*_pad_shape(array.shape))
    # print(f"array: {array.shape} @0x{array.ctypes.data_as(ctypes.c_void_p)}")
    # print(f"tensor_p: {shape(tensor_p)} @0x{tensor_p.contents.data:x}")

    # prevent the underlying numpy array to be freed
    setattr(tensor_p, "__data", array)
    return tensor_p


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


def UnityModel() -> NativeObj:
    return NativeObj("unity_model")


def GptVocab() -> NativeObj:
    return NativeObj("gpt_vocab")


def Fairseq2Model() -> NativeObj:
    return NativeObj("fairseq2_model")

lib.std_string_alloc.argtypes = [ctypes.c_char_p]
lib.std_string_alloc.restype = ctypes.c_void_p
lib.std_string_free.argtypes = [ctypes.c_void_p]
lib.std_string_free.restype = None
NativeObj._cache["std_string"] = (lib.std_string_alloc, lib.std_string_free)

@functools.lru_cache(1024)
def CppStr(content: str) -> NativeObj:
    c_str = ctypes.create_string_buffer(content.encode("utf-8"))
    cpp_str = lib.std_string_alloc(c_str)
    return NativeObj("std_string", cpp_str)


lib.unity_model_load.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p]


def unity_model_load(model_file: Path) -> Tuple[NativeObj, NativeObj]:
    model = UnityModel()
    vocab = GptVocab()
    lib.unity_model_load(
        ctypes.create_string_buffer(str(model_file).encode("utf-8")),
        model.ptr,
        vocab.ptr,
    )
    return model, vocab


lib.load_unity_ggml_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.load_unity_ggml_file.restype = None


def load_unity_ggml_file(model_file: Path) -> NativeObj:
    model = Fairseq2Model()
    bytes_file = ctypes.create_string_buffer(str(model_file).encode("utf-8"))
    lib.load_unity_ggml_file(model.ptr, bytes_file)
    return model


lib.unity_audio_encoder_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.unity_audio_encoder_graph.restype = ctypes.POINTER(ggml_cgraph)


def unity_audio_encoder_graph(model: NativeObj, tensor: ggml_tensor_p) -> ggml_cgraph_p:
    return lib.unity_audio_encoder_graph(model.ptr, tensor)  # type: ignore


lib.unity_eval.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ggml_tensor),
    ctypes.c_int,
]
lib.unity_eval.restype = ctypes.POINTER(ggml_cgraph)


def unity_eval(
    allocr: ctypes.c_void_p, model: NativeObj, tensor: ggml_tensor_p, n_threads: int
) -> ggml_cgraph_p:
    return lib.unity_eval(allocr, model.ptr, tensor, n_threads)


_FORWARD_CACHE: Dict[str, Callable[[...], ggml_tensor_p]] = {}


def forward(
    layer_name: str, model: NativeObj, prefix: str, *inputs: ggml_tensor_p
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
        return fwd(model.ptr, std_prefix, *inputs)  # ignore: type[no-any-return]
