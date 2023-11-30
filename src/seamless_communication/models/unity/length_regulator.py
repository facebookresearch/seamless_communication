# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from typing import Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.padding import PaddingMask, apply_padding_mask, to_padding_mask
from fairseq2.nn.projection import Linear
from fairseq2.nn.transformer import create_standard_layer_norm
from fairseq2.typing import DataType, Device
from torch import Tensor
from torch.nn import Conv1d, Dropout, Module, ReLU, Sequential

from seamless_communication.models.unity.film import FiLM


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
            upsampled_seqs[b, : upsampled_seq_lens[b]] = seqs[b].repeat_interleave(
                durations[b], dim=0
            )

        return upsampled_seqs, upsampled_seq_lens


class GaussianUpsampling(Module):
    """Gaussian upsampling with fixed temperature as in:
    https://arxiv.org/abs/2010.04301
    """

    def __init__(self, delta: float = 0.1):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        x: Tensor,
        durations: Tensor,
        padding_mask: Optional[PaddingMask] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Upsample hidden states according to durations.
        Args:
            x (Tensor): Batched hidden state to be expanded (B, T_text, C).
            durations (Tensor): Batched token duration (B, T_text).
            padding_mask (Tensor): Mask tensor (B, T_text).
        Returns:
            Tensor: Expanded hidden state (B, T_feat, C).
            Tensor: Output lengths (B,).
        """
        out_lens = durations.sum(dim=1)
        y_mask = to_padding_mask(out_lens, max(out_lens))

        B = durations.size(0)
        if durations.sum() == 0:
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            durations[durations.sum(dim=1).eq(0)] = 1

        if y_mask is None:
            T_feat = durations.sum().int()
        else:
            T_feat = y_mask.size(-1)

        t = torch.arange(0, T_feat).unsqueeze(0).repeat(B, 1).to(x)
        if y_mask is not None:
            t = t * y_mask.float()

        c = durations.cumsum(dim=-1) - durations / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2

        if padding_mask is not None:
            energy = energy.masked_fill(
                ~padding_mask.materialize().unsqueeze(1).repeat(1, T_feat, 1),
                -float("inf"),
            )

        p_attn = F.softmax(energy, dim=2).to(x)  # (B, T_feat, T_text)
        x = torch.matmul(p_attn, x)
        return x, out_lens


class VariancePredictor(Module):
    """Represents the duration/pitch/energy predictor as described in
    :cite:t:`https://arxiv.org/pdf/2006.04558.pdf`"""

    conv1: Sequential
    ln1: LayerNorm
    dropout_module: Dropout
    conv2: Sequential
    ln2: LayerNorm
    proj: Linear
    film: Optional[FiLM]

    def __init__(
        self,
        encoder_embed_dim: int,
        var_pred_hidden_dim: int,
        var_pred_kernel_size: int,
        var_pred_dropout: float,
        bias: bool = True,
        use_film: bool = False,
        film_cond_dim: int = 512,
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
                padding="same",
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
        )

        layer_norm_factory = create_standard_layer_norm

        self.ln1 = layer_norm_factory(var_pred_hidden_dim, device=device, dtype=dtype)

        self.dropout_module = Dropout(p=var_pred_dropout)

        self.conv2 = Sequential(
            Conv1d(
                var_pred_hidden_dim,
                var_pred_hidden_dim,
                var_pred_kernel_size,
                stride=1,
                padding="same",
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            ReLU(),
        )

        self.ln2 = layer_norm_factory(var_pred_hidden_dim, device=device, dtype=dtype)

        self.proj = Linear(
            var_pred_hidden_dim, 1, bias=True, device=device, dtype=dtype
        )

        if use_film:
            self.film = FiLM(
                film_cond_dim, var_pred_hidden_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("film", None)

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask] = None,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tensor:
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

        seqs = apply_padding_mask(seqs, padding_mask)

        if self.film is not None and film_cond_emb is not None:
            seqs = self.film(seqs, film_cond_emb)
            seqs = apply_padding_mask(seqs, padding_mask)

        # (N, S, H) -> (N, S, 1) -> (N, S)
        seqs = self.proj(seqs).squeeze(dim=2)

        return seqs


class VarianceAdaptor(Module):
    """Represent the Variance adaptor as described in
    :cite:t:`https://arxiv.org/pdf/2006.04558.pdf`"""

    duration_predictor: Optional[VariancePredictor]
    pitch_predictor: Optional[VariancePredictor]
    vuv_predictor: Optional[VariancePredictor]
    energy_predictor: Optional[VariancePredictor]
    length_regulator: Union[HardUpsampling, GaussianUpsampling]

    def __init__(
        self,
        duration_predictor: Optional[VariancePredictor] = None,
        pitch_predictor: Optional[VariancePredictor] = None,
        embed_pitch: Optional[Conv1d] = None,
        vuv_predictor: Optional[VariancePredictor] = None,
        energy_predictor: Optional[VariancePredictor] = None,
        embed_energy: Optional[Conv1d] = None,
        add_variance_parallel: bool = True,
        upsampling_type: Literal["gaussian", "hard"] = "hard",
    ):
        super().__init__()

        if duration_predictor:
            self.duration_predictor = duration_predictor
        else:
            self.register_module("duration_predictor", None)

        if pitch_predictor:
            self.pitch_predictor = pitch_predictor
            self.embed_pitch = embed_pitch
        else:
            self.register_module("pitch_predictor", None)
            self.register_module("embed_pitch", None)

        if vuv_predictor:
            self.vuv_predictor = vuv_predictor
        else:
            self.register_module("vuv_predictor", None)

        if energy_predictor:
            self.energy_predictor = energy_predictor
            self.embed_energy = embed_energy
        else:
            self.register_module("energy_predictor", None)
            self.register_module("embed_energy", None)

        self.add_variance_parallel = add_variance_parallel

        if upsampling_type == "gaussian":
            self.length_regulator = GaussianUpsampling()
        else:
            self.length_regulator = HardUpsampling()

    def forward(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        durations: Optional[Tensor] = None,
        duration_factor: float = 1.0,
        min_duration: int = 0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, PaddingMask, Tensor]:
        if self.duration_predictor is not None:
            log_durations = self.duration_predictor(seqs, padding_mask, film_cond_emb)
            durations = torch.clamp(
                torch.round((torch.exp(log_durations) - 1) * duration_factor).long(),
                min=min_duration,
            )
            # We need to apply the padding_mask again since we clamp by min_duration.
            durations = apply_padding_mask(durations, padding_mask, pad_value=0)

        assert durations is not None

        if self.pitch_predictor is not None:
            pitch_out = self.pitch_predictor(seqs, padding_mask, film_cond_emb)
            if self.vuv_predictor is not None:
                vuv_out = self.vuv_predictor(seqs, padding_mask, film_cond_emb)
                pitch_out = pitch_out * (torch.sigmoid(vuv_out) >= 0.5)

            assert self.embed_pitch is not None
            pitch_embed = self.embed_pitch(pitch_out.unsqueeze(1)).transpose(1, 2)
            if not self.add_variance_parallel:
                seqs = seqs + pitch_embed

        if self.energy_predictor is not None:
            energy_out = self.energy_predictor(seqs, padding_mask, film_cond_emb)

            assert self.embed_energy is not None
            energy_embed = self.embed_energy(energy_out.unsqueeze(1)).transpose(1, 2)
            if self.add_variance_parallel:
                seqs = seqs + pitch_embed + energy_embed
            else:
                seqs = seqs + energy_embed

        if isinstance(self.length_regulator, GaussianUpsampling):
            seqs, seq_lens = self.length_regulator(seqs, durations, padding_mask)
        else:
            seqs, seq_lens = self.length_regulator(seqs, durations)

        return seqs, PaddingMask(seq_lens, batch_seq_len=seqs.size(1)), durations
