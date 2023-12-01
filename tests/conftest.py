# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import tempfile
from argparse import ArgumentTypeError
from typing import cast
from urllib.request import urlretrieve

import pytest
import torch
from fairseq2.data.audio import AudioDecoder, AudioDecoderOutput
from fairseq2.memory import MemoryBlock
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


@pytest.fixture(scope="module")
def example_rate16k_audio() -> AudioDecoderOutput:
    url = "https://dl.fbaipublicfiles.com/seamlessM4T/LJ037-0171_sr16k.wav"

    audio_decoder = AudioDecoder(dtype=torch.float32, device=tests.common.device)

    with tempfile.NamedTemporaryFile() as f:
        urlretrieve(url, f.name)
        with open(f.name, "rb") as fb:
            block = MemoryBlock(fb.read())
        decoded_audio = audio_decoder(block)

    return decoded_audio
