# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum
from typing import Any, Tuple


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
        return ('?', value)
    if isinstance(value, Enum):
        return ('i', value.value)

    raise ValueError(f"Unsupported type {type(value)}")


def get_cpp_type(value) -> str:
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
        return f"Error[{e}]"

    if ctype == "i":
        return "int32_t"
    if ctype == "f":
        return "float"
    if ctype == "?":
        return "bool"

    raise RuntimeError(
        f"Should not have reached this part."
        f"Missing cpp translation for {ctype}"
    )
