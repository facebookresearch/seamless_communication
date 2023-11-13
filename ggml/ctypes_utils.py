import inspect
import ctypes
import types
import functools
from typing import TypeVar
from typing import Generic

T = TypeVar("T")


class Ptr(Generic[T]):
    contents: T

    def __new__(cls):
        breakpoint()
        return ctypes.pointer()


def c_struct(cls):
    struct = types.new_class(cls.__name__, bases=(ctypes.Structure,))
    struct.__module__ = cls.__module__
    struct._fields_ = [
        (k, _py_type_to_ctype(v)) for k, v in cls.__annotations__.items()
    ]

    return struct


@functools.lru_cache(256)
def _py_type_to_ctype(t: type):
    if isinstance(t, str):
        raise ValueError(
            f"Type parsing of '{t}' isn't supported, you need to provide a real type annotation."
        )
    if t is None:
        return None
    if isinstance(t, type):
        if t.__module__ == "ctypes":
            return t
        if issubclass(t, ctypes.Structure):
            return t
        if issubclass(t, ctypes._Pointer):
            return t
    if t is int:
        return ctypes.c_int
    if t is float:
        return ctypes.c_float
    if t is bool:
        return ctypes.c_bool
    if t is str:
        return ctypes.c_char_p

    if getattr(t, "__origin__", None) is Ptr:
        pointee = _py_type_to_ctype(t.__args__[0])
        return ctypes.POINTER(pointee)

    return ctypes.c_void_p


def _c_fn(module, fn):
    c_fn = getattr(module, fn.__name__)
    annotations = fn.__annotations__
    if "return" not in annotations:
        raise ValueError("@c_fn decorator requires type annotations on the decorated function.")

    c_fn.argtypes = [
        _py_type_to_ctype(t) for k, t in fn.__annotations__.items() if k != "return"
    ]
    c_fn.restype = _py_type_to_ctype(fn.__annotations__["return"])

    @functools.wraps(fn)
    def actual_fn(*args, **kwargs):
        raw_res = c_fn(*args, **kwargs)
        return raw_res

    return actual_fn


def c_fn(module):
    return functools.partial(_c_fn, module)
