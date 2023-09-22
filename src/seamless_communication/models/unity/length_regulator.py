# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torch import Tensor
from torch.nn import Conv1d, Dropout, Module, ReLU, Sequential

from typing import Optional, Tuple

from fairseq2.typing import DataType, Device
from fairseq2.nn.transformer import create_default_layer_norm
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.ops import repeat_interleave
from fairseq2.nn.projection import Linear
from fairseq2.nn.utils.mask import apply_padding_mask


class HardUpsampling(Module):
    """Upsamples sequences in a deterministic way as governed by durations."""

    def forward(self, seqs: Tensor, durations: Tensor) -> Tuple[Tensor, Tensor]:
        # seqs: (N, S, M), durations: (N, S)
        if durations.dtype not in (torch.int16, torch.int32, torch.int64):
            raise TypeError("The durations tensor should have an integer dtype.")

        upsampled_seq_lens = durations.sum(dim=1)
        max_len = int(upsampled_seq_lens.max().item())
        N, _, M = seqs.shape
        upsampled_seqs = seqs.new_zeros((N, max_len, M))

        for b in range(N):
            upsampled_seqs[b, : upsampled_seq_lens[b]] = repeat_interleave(
                seqs[b], dim=0, repeat=durations[b]
            )

        return upsampled_seqs, upsampled_seq_lens


class VariancePredictor(Module):
    """Represents the duration/pitch/energy predictor as described in
    :cite:t:`https://arxiv.org/pdf/2006.04558.pdf`"""

    conv1: Sequential
    ln1: LayerNorm
    dropout_module: Dropout
    conv2: Sequential
    ln2: LayerNorm
    proj: Linear

    def __init__(
        self,
        encoder_embed_dim: int,
        var_pred_hidden_dim: int,
        var_pred_kernel_size: int,
        var_pred_dropout: float,
        bias: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.conv1 = Sequential(
            Conv1d(
                encoder_embed_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                stride=1,
                padding=(var_pred_kernel_size - 1) // 2,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
        )

        layer_norm_fn = create_default_layer_norm

        self.ln1 = layer_norm_fn(var_pred_hidden_dim, device=device, dtype=dtype)

        self.dropout_module = Dropout(p=var_pred_dropout)

        self.conv2 = Sequential(
            Conv1d(
                var_pred_hidden_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                stride=1,
                padding=1,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
        )

        self.ln2 = layer_norm_fn(var_pred_hidden_dim, device=device, dtype=dtype)

        self.proj = Linear(
            var_pred_hidden_dim, 1, bias=True, device=device, dtype=dtype
        )

    def forward(self, seqs: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        # Ensure that we do not leak padded positions in the convolution layer.
        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, M) -> (N, M, S)
        seqs = seqs.transpose(1, 2)

        # (N, M, S) -> (N, H, S)
        seqs = self.conv1(seqs)

        # (N, H, S) -> (N, S, H)
        seqs = seqs.transpose(1, 2)

        seqs = self.ln1(seqs)

        seqs = self.dropout_module(seqs)

        seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, H) -> (N, H, S)
        seqs = seqs.transpose(1, 2)

        # (N, H, S) -> (N, H, S)
        seqs = self.conv2(seqs)

        # (N, H, S) -> (N, S, H)
        seqs = seqs.transpose(1, 2)

        seqs = self.ln2(seqs)

        seqs = self.dropout_module(seqs)

        # (N, S, H) -> (N, S, 1) -> (N, S)
        seqs = self.proj(seqs).squeeze(dim=2)

        return seqs


class VarianceAdaptor(Module):
    """Represent the Variance adaptor as described in
    :cite:t:`https://arxiv.org/pdf/2006.04558.pdf`"""

    duration_predictor: VariancePredictor
    pitch_predictor: Optional[VariancePredictor]
    energy_predictor: Optional[VariancePredictor]
    hard_upsampling: HardUpsampling

    def __init__(
        self,
        duration_predictor: VariancePredictor,
        pitch_predictor: Optional[VariancePredictor] = None,
        energy_predictor: Optional[VariancePredictor] = None,
    ):
        super().__init__()

        self.duration_predictor = duration_predictor

        if pitch_predictor:
            self.pitch_predictor = pitch_predictor
        else:
            self.register_module("pitch_predictor", None)

        if energy_predictor:
            self.energy_predictor = energy_predictor
        else:
            self.register_module("energy_predictor", None)

        self.hard_upsampling = HardUpsampling()

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        duration_factor: float = 1.0,
        min_duration: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        log_durations = self.duration_predictor(seqs, padding_mask)

        durations = torch.clamp(
            torch.round((torch.exp(log_durations) - 1) * duration_factor).long(),
            min=min_duration,
        )

        # We need to apply the padding_mask again since we clamp by min_duration.
        durations = apply_padding_mask(durations, padding_mask)

        # TODO: Implement pitch, energy predictors.
        # TODO: Implement GaussianUpsampling.
        seqs, seq_lens = self.hard_upsampling(seqs, durations)

        return seqs, seq_lens
