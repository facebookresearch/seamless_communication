# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path
from shutil import rmtree
import subprocess as sp
import tempfile
import select
from typing import Union, Dict, Optional, Tuple, IO
from torch import Tensor
import torchaudio
import io
import sys
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

    def copy_process_streams(self, process: sp.Popen):
        def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
            assert stream is not None
            if isinstance(stream, io.BufferedIOBase):
                stream = stream.raw
            return stream

        p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
        stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
            p_stdout.fileno(): (p_stdout, sys.stdout),
            p_stderr.fileno(): (p_stderr, sys.stderr),
        }
        fds = list(stream_by_fd.keys())

        while fds:
            # `select` syscall will wait until one of the file descriptors has content.
            ready, _, _ = select.select(fds, [], [])
            for fd in ready:
                p_stream, std = stream_by_fd[fd]
                raw_buf = p_stream.read(2 ** 16)
                if not raw_buf:
                    fds.remove(fd)
                    continue
                buf = raw_buf.decode()
                std.write(buf)
                std.flush()

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
            p = sp.Popen(cmd + audio, stdout=sp.PIPE, stderr=sp.PIPE)
            self.copy_process_streams(p)
            p.wait()

            if p.returncode != 0:
                print("Command failed, something went wrong.")
                return None
            
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
