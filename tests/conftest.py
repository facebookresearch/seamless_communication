# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentTypeError
from typing import cast

import pytest
from fairseq2.typing import Device

import tests.common


def parse_device_arg(value: str) -> Device:
    try:
        return Device(value)
    except RuntimeError:
        raise ArgumentTypeError(f"'{value}' is not a valid device name.")


def pytest_addoption(parser: pytest.Parser) -> None:
    # fmt: off
    parser.addoption(
        "--device", default="cpu", type=parse_device_arg,
        help="device on which to run tests (default: %(default)s)",
    )
    # fmt: on


def pytest_sessionstart(session: pytest.Session) -> None:
    tests.common.device = cast(Device, session.config.getoption("device"))
