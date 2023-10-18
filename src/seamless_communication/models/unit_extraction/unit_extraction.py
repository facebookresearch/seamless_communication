# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union
from pathlib import Path
import torch
import torch.nn.functional as F

from itertools import groupby
from torch import Tensor, nn

from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.typing import DataType, Device

from seamless_communication.models.unit_extraction.wav2vec2_layer_output import (
    load_wav2vec2_model,
    Wav2Vec2LayerOutputModel,
)
from seamless_communication.models.unit_extraction.kmeans import KmeansModel
from seamless_communication.models.inference import Translator
from seamless_communication.models.vocoder import load_vocoder_model, Vocoder


class UnitExtractor(nn.Module):
    """Unit Extractor which converts raw audio into units."""

    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        kmeans_uri: str,
        device: Device,
        dtype: DataType = torch.float32,
    ):
        super().__init__()
        wav2vec2_model = Translator.load_model_for_inference(
            load_wav2vec2_model, model_name_or_card, device, dtype
        )
        assert isinstance(wav2vec2_model, Wav2Vec2Model)
        self.model = Wav2Vec2LayerOutputModel(wav2vec2_model)
        self.device = device
        self.decode_audio = AudioDecoder(dtype=torch.float32, device=device)
        self.collate = Collater(pad_idx=2, pad_to_multiple=2)
        self.kmeans_model = KmeansModel(kmeans_uri, device)

    @torch.inference_mode()
    def predict(
        self,
        audio: Union[str, Tensor],
        out_layer_idx: int,
        sample_rate: int = 16000,
    ) -> Tensor:
        if isinstance(audio, str):
            with Path(audio).open("rb") as fb:
                block = MemoryBlock(fb.read())
            decoded_audio = self.decode_audio(block)
        else:
            decoded_audio = {
                "waveform": audio,
                "sample_rate": sample_rate,
                "format": -1,
            }
        src = self.collate(decoded_audio)["waveform"]
        seqs, padding_mask = get_seqs_and_padding_mask(src)
        seqs = seqs.view(1, -1)
        seqs = F.layer_norm(seqs, seqs.shape)
        batch = SequenceBatch(seqs=seqs, padding_mask=padding_mask)
        features = self.model(batch, out_layer_idx).squeeze(0)
        units = self.kmeans_model(features)
        return units

    @staticmethod
    def resynthesize_audio(
        units: Tensor,
        src_lang: str,
        device: Device,
        vocoder_name: str = "vocoder_36langs",
    ) -> Tensor:
        def reduce_list(lst):
            return [key for key, _ in groupby(lst)]

        reduced_units = reduce_list(units.cpu().tolist())

        vocoder = Translator.load_model_for_inference(
            load_vocoder_model, vocoder_name, device, torch.float32
        )
        assert isinstance(vocoder, Vocoder)
        wav = vocoder(reduced_units, src_lang, spkr=-1, dur_prediction=True)
        return wav
