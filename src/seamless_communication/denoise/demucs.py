# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path
import subprocess as sp
import tempfile
from typing import Union
from torch import Tensor
import torchaudio
from fairseq2.memory import MemoryBlock

SAMPLING_RATE = 16000

class Demucs():
    def __init__(
            self, 
            sample_rate=SAMPLING_RATE,
            model="htdemucs", 
            two_stems=None,
            float32=False,
            int24=False):
        self.sample_rate = SAMPLING_RATE
        self.model = model
        self.two_stems = two_stems
        self.float32 = float32
        self.int24 = int24

    def run_command_with_temp_file(self, cmd):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
            result = sp.run(cmd, stdout=temp, stderr=temp, text=True)
            # If there was an error, print the content of the file
            if result.returncode != 0:
                temp.seek(0)
                print(temp.read())

    def denoise(self, audio: Union[str, Tensor]):

        if isinstance(audio, Tensor):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                torchaudio.save(temp_wav.name, audio, sample_rate=self.sample_rate)
                audio = temp_wav.name

        if not Path(audio).exists():
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

            audio = [str(audio)]

            print("Executing command:", " ".join(cmd))
            self.run_command_with_temp_file(cmd + audio)
            
            separated_files = list(Path(temp_dir + "/htdemucs/noisy").glob("*vocals.wav*"))
            if not separated_files:
                print("Separated vocals file not found.")
                return None

            waveform, sample_rate = torchaudio.load(separated_files[0])

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                torchaudio.save(temp_wav.name, waveform, sample_rate=sample_rate)
                audio = temp_wav.name

            with Path(audio).open("rb") as fb:
                block = MemoryBlock(fb.read()) 

            return block
