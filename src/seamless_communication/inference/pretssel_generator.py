# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater, SequenceData
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device
from torch import Tensor

from seamless_communication.inference.translator import BatchedSpeechOutput
from seamless_communication.models.pretssel import load_pretssel_model
from seamless_communication.models.unity import load_unity_unit_tokenizer
from seamless_communication.models.vocoder import load_mel_vocoder_model


class PretsselGenerator(nn.Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        vocoder_name_or_card: Union[str, AssetCard],
        pretssel_name_or_card: Union[str, AssetCard],
        device: Device,
        gcmvn_mean: Optional[Tensor] = None,
        gcmvn_std: Optional[Tensor] = None,
        dtype: DataType = torch.float16,
    ):
        super().__init__()
        # Load the model.
        if device == torch.device("cpu"):
            dtype = torch.float32

        self.device = device
        self.dtype = dtype

        # Load the vocoder.
        self.vocoder = load_mel_vocoder_model(
            vocoder_name_or_card,
            device=device,
            dtype=dtype,
        )
        self.vocoder.eval()

        self.pretssel_model = load_pretssel_model(
            pretssel_name_or_card,
            device=device,
            dtype=dtype,
        )
        self.pretssel_model.eval()

        self.unit_tokenizer = load_unity_unit_tokenizer(model_name_or_card)
        self.unit_collate = Collater(pad_value=self.unit_tokenizer.vocab_info.pad_idx)
        self.duration_collate = Collater(pad_value=0)

        self.gcmvn_mean = gcmvn_mean
        self.gcmvn_std = gcmvn_std

    def gcmvn_denormalize(self, x: Tensor) -> Tensor:
        if self.gcmvn_mean is None or self.gcmvn_std is None:
            return x

        # x: B x T x C
        assert (
            x.ndim == 3
            and x.shape[2] == self.gcmvn_mean.shape[0] == self.gcmvn_std.shape[0]
        )
        x = x * self.gcmvn_std.view(1, 1, -1).expand_as(x)
        return x + self.gcmvn_mean.view(1, 1, -1).expand_as(x)

    @torch.inference_mode()
    def predict(
        self,
        units: List[List[int]],
        tgt_lang: str,
        padding_mask: Optional[PaddingMask],
        gcmvn_fbank: Tensor,
        sample_rate: int = 16000,
    ) -> BatchedSpeechOutput:
        list_units, durations = [], []
        unit_eos_token = torch.tensor(
            [self.unit_tokenizer.vocab_info.eos_idx],
            device=self.device,
        )

        for i, u in enumerate(units):
            unit = torch.tensor(u).to(unit_eos_token)

            # adjust the control symbols for the embedding
            unit += 4
            unit = torch.cat([unit, unit_eos_token], dim=0)

            unit, duration = torch.unique_consecutive(unit, return_counts=True)

            # adjust for the last eos token
            duration[-1] = 0

            list_units.append(unit)
            durations.append(duration * 2)

        speech_units = self.unit_collate(list_units)
        durations = self.duration_collate(durations)["seqs"]

        units_tensor, unit_padding_mask = get_seqs_and_padding_mask(speech_units)

        mel_output = self.pretssel_model(
            units_tensor,
            unit_padding_mask,
            gcmvn_fbank,
            padding_mask,
            tgt_lang=tgt_lang,
            durations=durations,
        )

        mel_output = self.gcmvn_denormalize(mel_output)

        audio_wavs = []
        for i, mel_out in enumerate(mel_output):
            # TODO: Implement batched inference for vocoder.
            mel_out = mel_out[: durations[i].sum()]
            translated_audio_wav = self.vocoder(mel_out, normalize_before=True)
            audio_wavs.append(translated_audio_wav.view(1, -1))

        return BatchedSpeechOutput(
            units=units,
            audio_wavs=audio_wavs,
            sample_rate=sample_rate,
        )
