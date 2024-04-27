# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
import tempfile

SAMPLING_RATE = 16000

class Demucs():
    def __int__(
            self, 
            model="htdemucs", 
            two_stems=None, 
            float32=False,
            int24=False):
        self.sample_rate = SAMPLING_RATE
        self.model = model
        self.two_stems = two_stems
        self.float32 = float32
        self.int24 = int24

    def denoise(self, input_file):

        if not Path(input_file).exists():
            print("Input file does not exist.")
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = ["python3", "-m", "demucs.separate", "-o", temp_dir, "-n", self.model]
            if self.float32:
                cmd += ["--float32"]
            if self.int24:
                cmd += ["--int24"]
            if self.two_stems is not None:
                cmd += [f"--two-stems={self.two_stems}"]

            cmd.append(input_file)

            print("Executing command:", " ".join(cmd))
            p = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
            stdout, stderr = p.communicate()

            if p.returncode != 0:
                print("Command failed, something went wrong.")
                print("Error message:", stderr.decode())
                return None

            separated_files = list(Path(temp_dir).glob("*vocals*"))
            if not separated_files:
                print("Separated vocals file not found.")
                return None

            output_file = separated_files[0].with_suffix(".wav")
            cmd_convert = ["ffmpeg", "-i", 
                           str(separated_files[0]), "-ar", 
                           str(self.sample_rate), "-ac", "1", 
                           str(output_file)]
            print("Executing command:", " ".join(cmd_convert))
            p_convert = sp.Popen(cmd_convert, stdout=sp.PIPE, stderr=sp.PIPE)
            stdout_convert, stderr_convert = p_convert.communicate()

            if p_convert.returncode != 0:
                print("Conversion to WAV failed.")
                print("Error message:", stderr_convert.decode())
                return None

            return str(output_file)
    



