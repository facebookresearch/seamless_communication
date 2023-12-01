# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import List

import torch
from torch.nn import Module

from fairseq2.typing import DataType, Device

from fairseq2.assets import asset_store
from fairseq2.data import (
    Collater,
    SequenceData,
    VocabularyInfo,
)
from fairseq2.nn.padding import get_seqs_and_padding_mask

from seamless_communication.inference import BatchedSpeechOutput
from seamless_communication.models.generator.loader import load_pretssel_vocoder_model


class PretsselGenerator(Module):
    def __init__(
        self,
        pretssel_name_or_card: str,
        vocab_info: VocabularyInfo,
        device: Device,
        dtype: DataType = torch.float16,
    ):
        super().__init__()
        # Load the model.
        if device == torch.device("cpu"):
            dtype = torch.float32

        self.device = device
        self.dtype = dtype

        self.pretssel_model = load_pretssel_vocoder_model(
            pretssel_name_or_card,
            device=device,
            dtype=dtype,
        )
        self.pretssel_model.eval()

        vocoder_model_card = asset_store.retrieve_card(pretssel_name_or_card)
        self.output_sample_rate = vocoder_model_card.field("sample_rate").as_(int)

        self.vocab_info = vocab_info
        self.unit_collate = Collater(pad_value=vocab_info.pad_idx)
        self.duration_collate = Collater(pad_value=0)
        self.unit_eos_token = torch.tensor([vocab_info.eos_idx], device=device)

    @torch.inference_mode()
    def predict(
        self,
        units: List[List[int]],
        tgt_lang: str,
        prosody_encoder_input: SequenceData,
    ) -> BatchedSpeechOutput:

        units_batch, durations = [], []
        for u in units:
            unit = torch.tensor(u).to(self.unit_eos_token)

            # adjust the control symbols for the embedding
            unit += 4
            unit = torch.cat([unit, self.unit_eos_token], dim=0)

            unit, duration = torch.unique_consecutive(unit, return_counts=True)

            # adjust for the last eos token
            duration[-1] = 0

            units_batch.append(unit)
            durations.append(duration * 2)

        speech_units = self.unit_collate(units_batch)
        durations = self.duration_collate(durations)["seqs"]

        units_tensor, unit_padding_mask = get_seqs_and_padding_mask(speech_units)
        prosody_input_seqs, prosody_padding_mask = get_seqs_and_padding_mask(
            prosody_encoder_input
        )

        audio_wavs = self.pretssel_model(
            units_tensor,
            tgt_lang,
            prosody_input_seqs,
            padding_mask=unit_padding_mask,
            prosody_padding_mask=prosody_padding_mask,
            durations=durations,
        )
        return BatchedSpeechOutput(
            units=units,
            audio_wavs=audio_wavs,
            sample_rate=self.output_sample_rate,
        )
