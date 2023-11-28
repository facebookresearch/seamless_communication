import ctypes
import dataclasses
import functools
import inspect
import types
from typing import Any, Callable, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class Ptr(Generic[T], ctypes._Pointer):  # type: ignore
    contents: T

    def __new__(cls, x: T) -> "Ptr[T]":
        return ctypes.pointer(x)  # type: ignore


NULLPTR: Ptr[Any] = None  # type: ignore[assignment]


def c_struct(cls: Type[T]) -> Type[T]:
    struct = types.new_class(cls.__name__, bases=(ctypes.Structure,))
    struct.__module__ = cls.__module__
    struct._fields_ = [  # type: ignore
        (k, _py_type_to_ctype(v)) for k, v in cls.__annotations__.items()
    ]

    def nice_init(self: T, *args: Any, **kwargs: Any) -> None:
        dc = cls(*args, **kwargs)
        for k, _ in self._fields_:  # type: ignore
            setattr(self, k, getattr(dc, k))

    setattr(struct, "__init__", nice_init)
    return struct


@functools.lru_cache(256)
def _py_type_to_ctype(t: type) -> type:
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
    if t is bytes:
        return ctypes.c_char_p
    if t is str:
        raise ValueError("str type is't supported by ctypes ?")

    if getattr(t, "__origin__", None) is Ptr:
        pointee = _py_type_to_ctype(t.__args__[0])  # type: ignore
        return ctypes.POINTER(pointee)

    return ctypes.c_void_p


F = TypeVar("F", bound=Callable[..., Any])


def _c_fn(module: Any, fn: F) -> F:
    if callable(module):
        c_fn = module
    else:
        c_fn = getattr(module, fn.__name__)
    annotations = fn.__annotations__
    if "return" not in annotations:
        raise ValueError(
            "@c_fn decorator requires type annotations on the decorated function."
        )

    c_fn.argtypes = [
        _py_type_to_ctype(t) for k, t in fn.__annotations__.items() if k != "return"
    ]
    c_fn.restype = _py_type_to_ctype(fn.__annotations__["return"])

    @functools.wraps(fn)
    def actual_fn(*args, **kwargs):  # type: ignore
        raw_res = c_fn(*args, **kwargs)
        return raw_res

    return actual_fn  # type: ignore


def c_fn(module: Any) -> Callable[[F], F]:
    return functools.partial(_c_fn, module)
