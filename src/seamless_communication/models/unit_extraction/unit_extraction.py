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
from fairseq2.data import Collater
import torch.nn.functional as F
from fairseq2.data.typing import StringLike
from fairseq2.memory import MemoryBlock
from fairseq2.assets.card import AssetCard
from fairseq2.models.sequence import SequenceBatch
from seamless_communication.models.unit_extraction.wav2vec2_layer_output import (
    load_wav2vec2_layer_output_model,
    Wav2Vec2LayerOutputModel,
)
from seamless_communication.models.unit_extraction.kmeans import KmeansModel
from seamless_communication.models.inference import Translator


class UnitExtractor(nn.Module):
    """Unit Extractor which converts raw audio into units."""

    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        kmeans_uri: str,
        device: Device,
        layer: int = 35,
    ):
        super().__init__()
        self.model: Wav2Vec2LayerOutputModel = Translator.load_model_for_inference(
            load_wav2vec2_layer_output_model, model_name_or_card, device, torch.float32
        )
        self.device = device
        self.decode_audio = AudioDecoder(dtype=torch.float32, device=device)
        self.collate = Collater(pad_idx=2, pad_to_multiple=2)
        self.kmeans_model = KmeansModel(kmeans_uri, device)

    @torch.no_grad()
    def predict(
        self,
        audio: Union[str, torch.Tensor],
        out_layer_idx: int,
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
        src = self.collate(decoded_audio)["waveform"]
        x = src["seqs"]
        x = F.layer_norm(x, x.shape)
        x = x.view(1, -1)
        batch = SequenceBatch(seqs=x, seq_lens=src["seq_lens"])
        features = self.model(batch, out_layer_idx).squeeze(0)
        units = self.kmeans_model(features)
        return units


if __name__ == "__main__":
    kmeans_uri = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    audio = "/large_experiments/seamless/ust/data/TTS/vocoder_training/audio_wavs/multi_spkr/eng/eng_LJSpeech-1.1_0/LJ003-0001.wav"
    device = torch.device("cuda:1")
    unit_extractor = UnitExtractor("xlsr2_1b_v2", kmeans_uri, device=Device("cuda:0"))
    out_layer_number = 35
    units = unit_extractor.predict(audio, out_layer_number - 1)
    print(units.shape, units.dtype, units.device)
