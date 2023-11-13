# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Any, Generator, List, Optional, Union

import torch
from fairseq2.typing import Device
from torch import Tensor

# The default device that tests should use. Note that pytest can change it based
# on the provided command line arguments.
device = Device("cpu")


def assert_close(
    a: Tensor,
    b: Union[Tensor, List[Any]],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal within a tolerance."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, rtol=rtol, atol=atol)  # type: ignore[attr-defined]


def assert_equal(a: Tensor, b: Union[Tensor, List[Any]]) -> None:
    """Assert that ``a`` and ``b`` are element-wise equal."""
    if not isinstance(b, Tensor):
        b = torch.tensor(b, device=device, dtype=a.dtype)

    torch.testing.assert_close(a, b, rtol=0, atol=0)  # type: ignore[attr-defined]


def has_no_inf(a: Tensor) -> bool:
    """Return ``True`` if ``a`` has no positive or negative infinite element."""
    return not torch.any(torch.isinf(a))


def has_no_nan(a: Tensor) -> bool:
    """Return ``True`` if ``a`` has no NaN element."""
    return not torch.any(torch.isnan(a))


@contextmanager
def tmp_rng_seed(device: Device, seed: int = 0) -> Generator[None, None, None]:
    """Set a temporary manual RNG seed.

    The RNG is reset to its original state once the block is exited.
    """
    device = Device(device)

    if device.type == "cuda":
        devices = [device]
    else:
        devices = []

    with torch.random.fork_rng(devices):
        torch.manual_seed(seed)

        yield
