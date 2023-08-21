# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Union
from pathlib import Path
import torch

from fairseq2.typing import Device
from torch import Tensor, nn
from fairseq2.data.audio import AudioDecoder
from fairseq2.models.wav2vec2 import load_wav2vec2_model, Wav2Vec2Model
from fairseq2.data.typing import StringLike
from fairseq2.memory import MemoryBlock


class UnitExtractor(nn.Module):
    """Vocoder interface to run vocoder models through hub. Currently we only support unit vocoder"""

    def __init__(
        self,
        model: Wav2Vec2Model,
        device: Device,
        layer: int = 35,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.to(device=device)
        self.device = device
        self.decode_audio = AudioDecoder(dtype=torch.float32, device=device)

    def predict(
        self,
        audio: Union[str, torch.Tensor],
    ) -> Tuple[List[Tensor], int]:

        if isinstance(audio, str):
            with Path(audio).open("rb") as fb:
                block = MemoryBlock(fb.read())
            decoded_audio = self.decode_audio(block)
        else:
            decoded_audio = {
                "waveform": audio,
                "sample_rate": 16000.0,
                "format": -1,
            }


if __name__ == "__main__":
    audio = "/large_experiments/seamless/ust/data/TTS/vocoder_training/audio_wavs/multi_spkr/eng/eng_LJSpeech-1.1_0/LJ003-0001.wav"
    model = load_wav2vec2_model("xlsr_1b_v2")
    unit_extractor = UnitExtractor(model, device=Device("cuda:0"))
    wav, sr = unit_extractor.predict(audio)
