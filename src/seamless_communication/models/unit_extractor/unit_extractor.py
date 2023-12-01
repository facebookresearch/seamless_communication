# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder
from fairseq2.memory import MemoryBlock
from fairseq2.models.sequence import SequenceBatch
from fairseq2.models.wav2vec2 import Wav2Vec2Model, load_wav2vec2_model
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device
from torch import Tensor, nn

from seamless_communication.models.unit_extractor.kmeans import KmeansModel
from seamless_communication.models.unit_extractor.wav2vec2_layer_output import (
    Wav2Vec2LayerOutputModel,
)
from seamless_communication.models.vocoder import Vocoder, load_vocoder_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


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

        wav2vec2_model = load_wav2vec2_model(
            model_name_or_card, device=device, dtype=dtype
        )
        wav2vec2_model.eval()
        assert isinstance(wav2vec2_model, Wav2Vec2Model)
        self.model = Wav2Vec2LayerOutputModel(wav2vec2_model)
        self.decode_audio = AudioDecoder(dtype=torch.float32, device=device)
        self.collate = Collater(pad_value=2, pad_to_multiple=2)
        self.kmeans_model = KmeansModel(kmeans_uri, device, dtype)
        self.device = device
        self.dtype = dtype

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
            assert (
                sample_rate == decoded_audio["sample_rate"]
            ), f"Input audio must have {sample_rate} sampling rate"

        else:
            assert audio.dim() <= 2, "The audio tensor can't be more than 2 dimensions."
            if audio.dim() == 1:
                audio = audio.unsqueeze(1)
            elif audio.dim() == 2 and audio.size(0) < audio.size(1):
                logger.warning(
                    "Transposing audio tensor from (bsz, seq_len) -> (seq_len, bsz)."
                )
                audio = audio.transpose(0, 1)

            decoded_audio = {
                "waveform": audio.to(dtype=self.dtype),
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
        return units  # type: ignore[no-any-return]

    @staticmethod
    def resynthesize_audio(
        units: Tensor,
        src_lang: str,
        device: Device,
        dtype: DataType,
        vocoder_name: str = "vocoder_v2",
    ) -> Tensor:
        vocoder = load_vocoder_model(vocoder_name, device=device, dtype=dtype)
        vocoder.eval()
        assert isinstance(vocoder, Vocoder)
        wav = vocoder(units, src_lang, spkr=-1, dur_prediction=True)
        return wav  # type: ignore[no-any-return]
