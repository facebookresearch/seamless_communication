# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq2.data import CString
from fairseq2.nn.embedding import StandardEmbedding
from fairseq2.nn.padding import to_padding_mask
from fairseq2.typing import DataType
from torch import Tensor
from torch.nn import Module

from seamless_communication.models.unity.char_tokenizer import CharTokenizer
from seamless_communication.models.unity.unit_tokenizer import UnitTokenizer


class UnitY2AlignmentFrontend(Module):
    def __init__(
        self,
        embed_text: StandardEmbedding,
        embed_unit: StandardEmbedding,
        text_tokenizer: CharTokenizer,
        unit_tokenizer: UnitTokenizer,
    ):
        super().__init__()
        self.embed_text = embed_text
        self.embed_unit = embed_unit
        self.text_tokenizer = text_tokenizer
        self.unit_tokenizer = unit_tokenizer
        unit_tokenizer.is_nar_decoder = True

        self.encode_text = self.text_tokenizer.create_raw_encoder()
        # text decoder can be used to map aligned characters to words
        self.decode_text = self.text_tokenizer.create_decoder()
        self.encode_unit = self.unit_tokenizer.create_encoder(lang="eng")

    def tokenize_text(
        self, text: str, return_tokens: bool = False, add_trailing_silence: bool = False
    ) -> Tensor:
        tokenized = self.encode_text(text)
        if add_trailing_silence:
            tokenized = torch.cat([tokenized, tokenized[0:1]])

        return tokenized

    def tokenize_text_to_tokens(
        self, text: str, add_trailing_silence: bool = False
    ) -> List[Union[CString, str]]:
        tokenized = self.encode_text.encode_as_tokens(text)
        if add_trailing_silence:
            tokenized = tokenized + [tokenized[0]]

        return tokenized

    def tokenize_unit(self, units: Union[str, Tensor]) -> Tensor:
        if isinstance(units, str):
            units = torch.tensor([int(u) for u in units.split(" ")])
        return self.encode_unit(units)

    def forward(self, text: Tensor, unit: Tensor) -> Tuple[Any, Any]:
        embs_unit = self.embed_unit(unit)
        embs_text = self.embed_text(text)
        return embs_text, embs_unit


class Permute12(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(1, 2)


class UnitY2AlignmentEncoder(Module):
    """
    UnitY2 Aligner component
    """

    def __init__(
        self,
        embed_dim: int,
        feat_dim: int,
        text_layers: int,
        feat_layers: int,
        dropout: float,
        temperature: float,
        reduction_factor: int,
        dtype: DataType,
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction_factor = reduction_factor  # for unit

        layers: List[Module] = [Permute12()]
        for i in range(text_layers):
            if i < text_layers - 1:
                layers.append(
                    nn.Conv1d(
                        embed_dim, embed_dim, kernel_size=3, padding=1, dtype=dtype
                    )
                )
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            else:
                layers.append(
                    nn.Conv1d(
                        embed_dim, embed_dim, kernel_size=1, padding=0, dtype=dtype
                    )
                )
                layers.append(nn.Dropout(p=dropout))
                layers.append(Permute12())
        self.t_conv = nn.Sequential(*layers)

        layers = [Permute12()]
        input_dim = feat_dim
        for i in range(feat_layers):
            if i < feat_layers - 1:
                layers.append(
                    nn.Conv1d(
                        input_dim, embed_dim, kernel_size=3, padding=1, dtype=dtype
                    )
                )
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
            else:
                layers.append(
                    nn.Conv1d(
                        input_dim,
                        embed_dim,
                        kernel_size=1,
                        padding=0,
                        stride=reduction_factor,
                        dtype=dtype,
                    )
                )
                layers.append(nn.Dropout(p=dropout))
                layers.append(Permute12())
            input_dim = embed_dim
        self.f_conv = nn.Sequential(*layers)

    def forward(
        self,
        text_emb: Tensor,
        feat_emb: Tensor,
        text_lengths: Tensor,
        feat_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute alignment between sequence of text and feature embeddings

        Args:
            text_emb (Tensor): Batched text embedding (B, T_text, C).
            feat_emb (Tensor): Batched acoustic feature (B, T_feat, feat_dim).
            text_lengths (Tensor): Source text length (B,).
            feat_lengths (Tensor): Target feature length (B,).

        Returns:
            Tensor: Log probability of attention matrix (B, T_feat, T_text)
            Tensor: Unit durations of every text token (B, T_text)

        """
        _feat_lengths = feat_lengths.clone()
        if self.reduction_factor > 1:
            feat_lengths = torch.ceil(feat_lengths / self.reduction_factor).long()

        text_emb = self.t_conv(text_emb)
        feat_emb = self.f_conv(feat_emb)

        dist = feat_emb.unsqueeze(2) - text_emb.unsqueeze(1)
        dist = torch.norm(dist, p=2, dim=3)
        score = -self.temperature * dist

        padding_mask = ~(to_padding_mask(text_lengths, max(text_lengths)))
        padding_mask = padding_mask.unsqueeze(-2)
        score = score.masked_fill(padding_mask, -np.inf)

        attn_lprob = F.log_softmax(score, dim=-1)

        attn_hard_dur = viterbi_decode(attn_lprob, text_lengths, feat_lengths)

        if self.reduction_factor > 1:
            attn_hard_dur = self.postprocess_alignment(
                attn_hard_dur, text_lengths, _feat_lengths
            )

        return attn_lprob, attn_hard_dur

    def postprocess_alignment(
        self, attn_hard_dur: Tensor, text_lengths: Tensor, feat_lengths: Tensor
    ) -> Tensor:
        attn_hard_dur = attn_hard_dur * self.reduction_factor
        B, T = attn_hard_dur.size()  # B x T_text
        dur_cumsum = torch.cumsum(attn_hard_dur, dim=1)
        for b in range(B):
            for t in range(text_lengths[b]):
                # truncate the right frames
                if dur_cumsum[b, t] >= feat_lengths[b]:
                    if t == 0:
                        attn_hard_dur[b, t] = feat_lengths[b]
                    else:
                        attn_hard_dur[b, t] = feat_lengths[b] - dur_cumsum[b, t - 1]
                    if t < text_lengths[b] - 1:
                        attn_hard_dur[b, t + 1 :] = 0
                    break
        return attn_hard_dur


def _monotonic_alignment_search(
    attn_lprob: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    # https://arxiv.org/abs/2005.11129
    T_feat = attn_lprob.shape[0]
    T_text = attn_lprob.shape[1]
    Q = np.full((T_text, T_feat), fill_value=-np.inf)

    log_prob = attn_lprob.transpose(1, 0)  # -> (T_text, T_feat)
    # 1.  Q <- init first row for all j
    for j in range(T_feat):
        Q[0, j] = log_prob[0, : j + 1].sum()

    # 2.
    for j in range(1, T_feat):
        for i in range(1, min(j + 1, T_text)):
            Q[i, j] = max(Q[i - 1, j - 1], Q[i, j - 1]) + log_prob[i, j]

    # 3.
    A = np.full((T_feat,), fill_value=T_text - 1)
    for j in range(T_feat - 2, -1, -1):  # T_feat-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j + 1] - 1
        i_b = A[j + 1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a, j] >= Q[i_b, j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    return A


def viterbi_decode(
    attn_lprob: Tensor, text_lengths: Tensor, feat_lengths: Tensor
) -> Tensor:
    """Extract duration from an attention probability matrix

    Args:
        attn_lprob (Tensor): Batched log probability of attention
            matrix (B, T_feat, T_text).
        text_lengths (Tensor): Text length tensor (B,).
        feat_lengths (Tensor): Feature length tensor (B,).

    Returns:
        Tensor: Batched token duration extracted from `attn_lprob` (B, T_text).
        Tensor: Binarization loss tensor ().

    """
    B = attn_lprob.size(0)
    T_text = attn_lprob.size(2)
    device = attn_lprob.device

    durations = torch.zeros((B, T_text), device=device, dtype=torch.long)
    for b in range(B):
        assert feat_lengths[b] > 0
        assert text_lengths[b] > 0
        cur_log_p_attn = attn_lprob[b, : feat_lengths[b], : text_lengths[b]]
        viterbi = _monotonic_alignment_search(
            cur_log_p_attn.float().detach().cpu().numpy()
        )
        _durations = np.bincount(viterbi)
        durations[b, : len(_durations)] = torch.from_numpy(_durations).to(device)

    return durations


class UnitY2AlignmentModel(Module):
    alignment_encoder: UnitY2AlignmentEncoder
    alignment_frontend: UnitY2AlignmentFrontend

    def __init__(
        self,
        alignment_frontend: UnitY2AlignmentFrontend,
        alignment_encoder: UnitY2AlignmentEncoder,
    ):
        super().__init__()
        self.alignment_frontend = alignment_frontend
        self.alignment_encoder = alignment_encoder

    def forward(self, input_text: Tensor, input_unit: Tensor) -> Tuple[Tensor, Tensor]:
        assert input_text.ndim == 2
        assert input_unit.ndim == 2
        embs_text, embs_unit = self.alignment_frontend(input_text, input_unit)
        attn_lprob, attn_hard_dur = self.alignment_encoder(
            embs_text,
            embs_unit,
            torch.tensor([embs_text.size(1)]).to(embs_text).int(),
            torch.tensor([embs_unit.size(1)]).to(embs_unit).int(),
        )

        return attn_lprob, attn_hard_dur
