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
from dataclasses import dataclass
from typing import Optional
import os
import logging

SAMPLING_RATE = 16000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger("demucs")

@dataclass
class DenoisingConfig:
    def __init__(
            self,
            filter_width: int = 3,
            model="htdemucs", 
            sample_rate=SAMPLING_RATE,
            two_stems=None,
            float32=False,
            int24=False):
        self.filter_width = filter_width
        self.model = model
        self.sample_rate = sample_rate
        self.two_stems = two_stems
        self.float32 = float32
        self.int24 = int24

class Demucs():
    def __init__(
            self, 
            denoise_config: Optional[DenoisingConfig]):
        self.denoise_config = denoise_config
        self.temp_files = []

    def run_command_with_temp_file(self, cmd):
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
            self.temp_files.append(temp.name)
            result = sp.run(cmd, stdout=temp, stderr=temp, text=True)
            # If there was an error, log the content of the file
            if result.returncode != 0:
                temp.seek(0)
                logger.info(temp.read())

    def cleanup_temp_files(self):
        for temp_file in self.temp_files:
            try:
                os.remove(temp_file)  
            except Exception as e:
                logger.info(f"Failed to remove temporary file: {temp_file}. Error: {e}")

    def denoise(self, audio: Union[str, Tensor]):

        if self.denoise_config is None:
          self.denoise_config = DenoisingConfig()

        if isinstance(audio, Tensor):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                self.temp_files.append(temp_wav.name)
                torchaudio.save(temp_wav.name, audio, self.denoise_config.sample_rate)
                audio = temp_wav.name

        if not Path(audio).exists():
            logger.info("Input file does not exist.")
            return None

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd = ["python3", "-m", "demucs.separate", "-o", temp_dir, "-n", self.denoise_config.model]
            if self.denoise_config.float32:
                cmd += ["--float32"]
            if self.denoise_config.int24:
                cmd += ["--int24"]
            if self.denoise_config.two_stems is not None:
                cmd += [f"--two-stems={self.denoise_config.two_stems}"]

            audio_path = Path(audio)
            audio_name = audio_path.stem
            audio = [str(audio)]

            logger.info("Executing command:", " ".join(cmd))
            self.run_command_with_temp_file(cmd + audio)

            separated_files = list(Path(temp_dir + "/htdemucs/" + audio_name).glob("*vocals.wav*"))
            
            if not separated_files:
                logger.info("Separated vocals file not found.")
                return None

            waveform, sample_rate = torchaudio.load(separated_files[0])

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav2:
                torchaudio.save(temp_wav2.name, waveform, sample_rate=sample_rate)
                block = MemoryBlock(temp_wav2.read())

            self.cleanup_temp_files()

            return block
        