# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Tuple, Union

import numpy
import torch
import torch.nn as nn
import torchaudio
from fairseq2.typing import DataType, Device
from fairseq2.data.typing import StringLike
from torch import Tensor

from seamless_communication.models.aligner.loader import load_unity2_alignment_model
from seamless_communication.models.unit_extractor import UnitExtractor

try:
    import matplotlib.pyplot as plt

    matplotlib_available = True
except ImportError:
    matplotlib_available = False


class AlignmentExtractor(nn.Module):
    def __init__(
        self,
        aligner_model_name_or_card: str,
        unit_extractor_model_name_or_card: Union[Any, str] = None,
        unit_extractor_output_layer: Union[Any, int] = None,
        unit_extractor_kmeans_model_uri: Union[Any, str] = None,
        device: Device = Device("cpu"),
        dtype: DataType = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        if self.dtype == torch.float16 and self.device == Device("cpu"):
            raise RuntimeError("FP16 only works on GPU, set args accordingly")

        self.alignment_model = load_unity2_alignment_model(
            aligner_model_name_or_card, device=self.device, dtype=self.dtype
        )
        self.alignment_model.eval()

        self.unit_extractor = None
        self.unit_extractor_output_layer = 0

        if unit_extractor_model_name_or_card is not None:
            self.unit_extractor = UnitExtractor(
                unit_extractor_model_name_or_card,
                unit_extractor_kmeans_model_uri,
                device=device,
                dtype=dtype,
            )
            self.unit_extractor_output_layer = unit_extractor_output_layer

    def load_audio(
        self, audio_path: str, sampling_rate: int = 16_000
    ) -> Tuple[Tensor, int]:
        assert os.path.exists(audio_path)
        audio, rate = torchaudio.load(audio_path)
        if rate != sampling_rate:
            audio = torchaudio.functional.resample(audio, rate, sampling_rate)
            rate = sampling_rate
        return audio, rate

    def prepare_audio(self, audio: Union[str, Tensor]) -> Tensor:
        # TODO: switch to fairseq2 data pipeline once it supports resampling
        if isinstance(audio, str):
            audio, _ = self.load_audio(audio, sampling_rate=16_000)
        if audio.ndim > 1:
            # averaging over channels
            assert audio.size(0) < audio.size(
                1
            ), "Expected [Channel,Time] shape, but Channel > Time"
            audio = audio.mean(0)
        assert (
            audio.ndim == 1
        ), f"After channel averaging audio shape expected to be [Time] i.e. mono audio"
        audio = audio.to(self.device, self.dtype)

        return audio

    def extract_units(self, audio: Tensor) -> Tensor:
        assert isinstance(
            self.unit_extractor, UnitExtractor
        ), "Unit extractor is required to get units from audio tensor"
        units = self.unit_extractor.predict(audio, self.unit_extractor_output_layer)
        return units

    @torch.inference_mode()
    def extract_alignment(
        self,
        audio: Union[str, Tensor],
        text: str,
        plot: bool = False,
        add_trailing_silence: bool = False,
    ) -> Tuple[Tensor, Tensor, List[StringLike]]:
        if isinstance(audio, Tensor) and not torch.is_floating_point(audio):
            # we got units as audio arg
            units = audio
            units = units.to(self.device)
            audio_tensor = None
        else:
            audio_tensor = self.prepare_audio(audio)
            units = self.extract_units(audio_tensor)

        tokenized_unit_ids = self.alignment_model.alignment_frontend.tokenize_unit(
            units
        ).unsqueeze(0)
        tokenized_text_ids = (
            self.alignment_model.alignment_frontend.tokenize_text(
                text, add_trailing_silence=add_trailing_silence
            )
            .to(self.device)
            .unsqueeze(0)
        )
        tokenized_text_tokens = (
            self.alignment_model.alignment_frontend.tokenize_text_to_tokens(
                text, add_trailing_silence=add_trailing_silence
            )
        )
        _, alignment_durations = self.alignment_model(
            tokenized_text_ids, tokenized_unit_ids
        )

        if plot and (audio_tensor is not None):
            self.plot_alignment(
                audio_tensor.cpu(), tokenized_text_tokens, alignment_durations.cpu()
            )

        return alignment_durations, tokenized_text_ids, tokenized_text_tokens

    def detokenize_text(self, tokenized_text_ids: Tensor) -> StringLike:
        return self.alignment_model.alignment_frontend.decode_text(tokenized_text_ids)

    def plot_alignment(
        self, audio: Tensor, text_tokens: List[StringLike], durations: Tensor
    ) -> None:
        if not matplotlib_available:
            raise RuntimeError(
                "Please `pip install matplotlib` in order to use plot alignment."
            )
        _, ax = plt.subplots(figsize=(22, 3.5))
        ax.plot(audio, color="gray", linewidth=0.3)
        durations_cumul = numpy.concatenate([numpy.array([0]), numpy.cumsum(durations)])
        alignment_ticks = durations_cumul * 320  # 320 is hardcoded for 20ms rate here

        ax.vlines(
            alignment_ticks,
            ymax=1,
            ymin=-1,
            color="indigo",
            linestyles="dashed",
            lw=0.5,
        )

        middle_tick_positions = (
            durations_cumul[:-1] + (durations_cumul[1:] - durations_cumul[:-1]) / 2
        )
        ax.set_xticks(middle_tick_positions * 320)
        ax.set_xticklabels(text_tokens, fontsize=13)
        ax.set_xlim(0, len(audio))

        ax.set_ylim(audio.min(), audio.max())
        ax.set_yticks([])
        plt.tight_layout()
        plt.show()
