# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import struct
from io import BufferedWriter

import torch

from ggml.examples.unity.type_utils import to_ctype


class BufferedGGMLWriter:
    buffer: BufferedWriter

    def __init__(self, buffer: BufferedWriter) -> None:
        self.buffer = buffer

    def write_magic_hex(self) -> None:
        """Write GGML Magic Number to internal buffer.
        This should be called at the start of your convert process.
        """
        self.buffer.write(struct.pack("i", 0x67676d6c))

    def write_hparams(self, hparams: dict) -> None:
        """Write hyper parameters to internal buffer.

        :params hparams:
            flattened dict containing model's hyper parameters.
        """
        for key in hparams.keys():
            try:
                value = hparams[key]
                ctype, cvalue = to_ctype(value)
                self.buffer.write(struct.pack(ctype, cvalue))
            except ValueError as e:
                # TODO use logger
                print(f"[Warning] {e}. Skipping config for key {key}")
                continue

    def write_state_dict(self, state_dict: dict) -> None:
        """Write pytorch state dict to internal buffer.

        :paras state_dict:
            state dict returned by pytorch model
        """
        for key, value in state_dict.items():
            self.write_string(key)
            self.write_tensor(value)

    def write_string(self, value: str) -> None:
        """Write string in utf-8 format to internal buffer.

        :params value:
            string value to dump.
        """
        str_ = value.encode("utf-8")
        self.buffer.write(struct.pack("i", len(str_)))
        self.buffer.write(str_)

    def write_tensor(self, value: torch.Tensor) -> None:
        """Write torch tensor in ggml format to internal buffer.

        First we save the number of dimensions and the dtype.
        Then we save the data as numpy array.

        :params value:
            Tensor to dump.
        """
        data = value.squeeze().numpy()
        n_dims = len(data.shape)

        # TODO: Convert to fp16 when necessary!
        ftype = 0

        self.buffer.write(struct.pack("ii", n_dims, ftype))
        for i in range(n_dims):
            self.buffer.write(struct.pack("i", data.shape[n_dims - 1 - i]))

        data.tofile(self.buffer)
