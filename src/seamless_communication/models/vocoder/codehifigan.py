# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from seamless_communication.models.unity import VariancePredictor
from seamless_communication.models.vocoder.hifigan import Generator


class CodeGenerator(Generator):
    def __init__(
        self,
        upsample_rates: List[int],
        upsample_kernel_sizes: List[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        model_in_dim: Optional[int],
        num_embeddings: int,
        embedding_dim: int,
        dur_predictor_params: Dict[str, Any],
        lang_embedding_dim: int,
        num_langs: int,
        spkr_embedding_dim: int,
        num_spkrs: int,
    ):
        super().__init__(
            upsample_rates,
            upsample_kernel_sizes,
            upsample_initial_channel,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            model_in_dim,
        )
        self.dict = nn.Embedding(num_embeddings, embedding_dim)
        self.spkr = nn.Embedding(num_spkrs, spkr_embedding_dim)
        self.lang = nn.Embedding(num_langs, lang_embedding_dim)

        self.dur_predictor = None
        if dur_predictor_params:
            self.dur_predictor = VariancePredictor(**dur_predictor_params)

        self.num_spkrs = num_spkrs
        self.num_langs = num_langs

    @staticmethod
    def _upsample(signal: Tensor, max_frames: int) -> Tensor:
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, sample: Dict[str, Any], dur_prediction: bool) -> Tensor:  # type: ignore
        x = sample["code"]
        x = self.dict(x).transpose(1, 2)

        if self.dur_predictor and dur_prediction:
            log_dur_pred = self.dur_predictor(x.transpose(1, 2), None)
            dur_out = torch.clamp(
                torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
            )
            # B x C x T
            repeat_interleaved_x = []
            for i in range(x.size(0)):
                repeat_interleaved_x.append(torch.repeat_interleave(x[i].unsqueeze(0), dur_out[i].view(-1), dim=2))
            x = torch.cat(repeat_interleaved_x)
        upsampled_spkr = []
        upsampled_lang = []
        spkr = self.spkr(sample["spkr"]).transpose(1, 2)
        lang = self.lang(sample["lang"]).transpose(1, 2)
        for i in range(x.size(0)):
            upsampled_spkr.append(self._upsample(spkr[i], x.shape[-1]))
            upsampled_lang.append(self._upsample(lang[i], x.shape[-1]))
        spkr = torch.cat(upsampled_spkr, dim=1).transpose(0, 1)
        lang = torch.cat(upsampled_lang, dim=1).transpose(0, 1)
        x = torch.cat([x, spkr], dim=1)
        x = torch.cat([lang, x], dim=1)

        return super().forward(x)
