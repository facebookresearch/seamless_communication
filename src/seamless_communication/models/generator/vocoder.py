# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq2.nn.embedding import Embedding, StandardEmbedding
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.projection import Projection
from fairseq2.typing import DataType, Device
from torch.nn import (
    ELU,
    BatchNorm1d,
    Conv1d,
    ConvTranspose1d,
    Dropout,
    Module,
    ModuleList,
    Parameter,
    Sequential,
    Tanh,
    init,
)
from torch.nn.utils.weight_norm import remove_weight_norm, weight_norm

from seamless_communication.models.generator.ecapa_tdnn import ECAPA_TDNN
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.length_regulator import VarianceAdaptor
from seamless_communication.models.vocoder.hifigan import (
    LRELU_SLOPE,
    ResBlock,
    init_weights,
)

from .streamable import (
    StreamableConv1d,
    StreamableConvTranspose1d,
    StreamableLSTM,
    StreamableResnetBlock,
)

ELU_PARAMS: Dict[str, Any] = {"alpha": 1.0}


class PretsselEncoderFrontend(Module):
    """
    Represent Encoder frontend, including the prosody encoder and language embedding
    """

    prosody_encoder: ECAPA_TDNN
    embed_tokens: Embedding
    embed_positions: PositionEncoder
    pos_emb_alpha: Parameter
    embed_lang: Embedding
    dropout: Dropout

    def __init__(
        self,
        prosody_encoder: ECAPA_TDNN,
        embed_tokens: Embedding,
        embed_positions: PositionEncoder,
        lang_to_index: Dict[str, int],
        lang_embed_dim: Optional[int],
        dropout_p: float,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.prosody_encoder = prosody_encoder

        self.embed_tokens = embed_tokens

        self.embed_positions = embed_positions
        self.pos_emb_alpha = Parameter(torch.ones(1, device=device, dtype=dtype))

        self.lang_to_index = lang_to_index

        if lang_embed_dim is not None:
            self.embed_lang = StandardEmbedding(
                len(lang_to_index), lang_embed_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("embed_lang", None)

        self.dropout = Dropout(dropout_p)

        self.device = device
        self.dtype = dtype

    def forward(
        self,
        seqs: torch.Tensor,
        padding_mask: Optional[PaddingMask],
        prosody_input_seqs: torch.Tensor,
        prosody_padding_mask: Optional[PaddingMask],
        tgt_lang: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prosody_embs = self.prosody_encoder(
            prosody_input_seqs,
            prosody_padding_mask,
        ).unsqueeze(1)

        if self.embed_lang is not None:
            lang_index = self.lang_to_index[tgt_lang]
            lang_index_tensor = (
                torch.Tensor([lang_index]).to(seqs).repeat(seqs.size(0), 1)
            )
            lang_embeds = self.embed_lang(lang_index_tensor)
            prosody_embs = torch.cat([prosody_embs, lang_embeds], dim=-1)

        seqs = self.embed_tokens(seqs)
        seqs += self.pos_emb_alpha * (self.embed_positions(seqs, padding_mask) - seqs)
        seqs = self.dropout(seqs)

        return seqs, prosody_embs


class PretsselDecoderFrontend(Module):
    """Represent Decoder frontend, including VarianceAdaptor & Positional embedding"""

    variance_adaptor: VarianceAdaptor
    embed_positions: PositionEncoder
    pos_emb_alpha: Parameter

    def __init__(
        self,
        variance_adaptor: VarianceAdaptor,
        embed_positions: PositionEncoder,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()

        self.variance_adaptor = variance_adaptor
        self.embed_positions = embed_positions
        self.pos_emb_alpha = Parameter(torch.ones(1, device=device, dtype=dtype))

        self.device = device
        self.dtype = dtype

    def forward(
        self,
        seqs: torch.Tensor,
        padding_mask: PaddingMask,
        durations: Optional[torch.Tensor] = None,
        duration_factor: float = 1.0,
        min_duration: int = 0,
        film_cond_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, PaddingMask]:
        seqs, padding_mask, _ = self.variance_adaptor(
            seqs, padding_mask, durations, duration_factor, min_duration, film_cond_emb
        )

        seqs += self.pos_emb_alpha * (self.embed_positions(seqs, padding_mask) - seqs)

        return seqs, padding_mask


class PretsselVocoder(Module):
    """The expressivity-preserving vocoder"""

    encoder_frontend: PretsselEncoderFrontend
    encoder: FeedForwardTransformer
    decoder_frontend: PretsselDecoderFrontend
    decoder: FeedForwardTransformer
    final_proj: Projection

    def __init__(  # type: ignore[no-untyped-def]
        self,
        encoder_frontend: PretsselEncoderFrontend,
        encoder: FeedForwardTransformer,
        decoder_frontend: PretsselDecoderFrontend,
        decoder: FeedForwardTransformer,
        final_proj: Projection,
        pn_n_channels: int,
        pn_kernel_size: int,
        pn_layers: int,
        pn_dropout: float,
        upsample_rates: List[int],
        upsample_kernel_sizes: List[int],
        upsample_initial_channel: int,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        mel_dim: int = 80,
        add_ups_out_pad: bool = True,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        ratios: List[int] = [8, 5, 4, 2],
        norm: Literal[
            "none", "weight_norm", "spectral_norm", "time_group_norm"
        ] = "none",
        norm_params: Dict[str, Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        causal: bool = False,
        pad_mode: str = "constant",
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 0,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
        gcmvn_mean: Optional[List[float]] = None,
        gcmvn_std: Optional[List[float]] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        mult = 1
        stream_layers: List[Module] = [
            StreamableConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm="none" if disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                activation=Tanh(),
                device=device,
                dtype=dtype,
            )
        ]
        # Downsample to from audio scale
        for i, ratio in enumerate(list(reversed(ratios))):
            block_norm = "none" if disable_norm_outer_blocks >= i + 2 else norm
            stream_layers.append(
                StreamableResnetBlock(
                    mult * n_filters,
                    kernel_sizes=[residual_kernel_size, 1],
                    dilations=[1, 1],
                    norm=block_norm,
                    norm_params=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    compress=compress,
                    true_skip=true_skip,
                    device=device,
                    dtype=dtype,
                )
            )
            stream_layers.append(ELU(**ELU_PARAMS))
            stream_layers.append(
                StreamableConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    device=device,
                    dtype=dtype,
                )
            )
            mult *= 2

        stream_layers.append(StreamableLSTM(mult * n_filters, num_layers=lstm))
        stream_layers.append(ELU(**ELU_PARAMS))
        n_blocks = len(ratios) + 2
        stream_layers.append(
            StreamableConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm="none" if disable_norm_outer_blocks == n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                device=device,
                dtype=dtype,
            )
        )
        stream_layers.append(
            StreamableConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm="none" if disable_norm_outer_blocks == n_blocks else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                device=device,
                dtype=dtype,
            )
        )
        stream_layers.append(
            StreamableLSTM(
                mult * n_filters, num_layers=lstm, device=device, dtype=dtype
            )
        )

        # resample back to raw audio scale
        for i, ratio in enumerate(ratios):
            block_norm = (
                "none" if disable_norm_outer_blocks >= n_blocks - (i + 1) else norm
            )
            stream_layers.append(ELU(**ELU_PARAMS))
            stream_layers.append(
                StreamableConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio,
                    device=device,
                    dtype=dtype,
                )
            )
            stream_layers.append(
                StreamableResnetBlock(
                    mult * n_filters // 2,
                    kernel_sizes=[residual_kernel_size, 1],
                    dilations=[1, 1],
                    norm=block_norm,
                    norm_params=norm_params,
                    activation_params=ELU_PARAMS,
                    causal=causal,
                    pad_mode=pad_mode,
                    compress=compress,
                    true_skip=true_skip,
                    device=device,
                    dtype=dtype,
                )
            )
            mult //= 2

        stream_layers.append(ELU(**ELU_PARAMS))
        stream_layers.append(
            StreamableConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                device=device,
                dtype=dtype,
            )
        )
        self.n_streams = len(stream_layers)
        chunk_size = self.n_streams // 4
        stream_idx = 0

        self.pn_layers = pn_layers
        self.layers = ModuleList()
        assert pn_kernel_size % 2 == 1
        for i in range(pn_layers):
            cur_layers = (
                [
                    Conv1d(
                        mel_dim if i == 0 else pn_n_channels,
                        pn_n_channels if i < pn_layers - 1 else mel_dim,
                        kernel_size=pn_kernel_size,
                        padding="same",
                        device=device,
                        dtype=dtype,
                    ),
                    BatchNorm1d(
                        pn_n_channels if i < pn_layers - 1 else mel_dim,
                        device=device,
                        dtype=dtype,
                    ),
                ]
                + ([Tanh()] if i < pn_layers - 1 else [])
                + [Dropout(pn_dropout)]
            )
            self.layers.append(Sequential(*cur_layers))
        self.reset_parameters()
        self.layers.extend(stream_layers[:chunk_size])
        stream_idx += chunk_size
        self.layers.append(
            weight_norm(
                Conv1d(
                    mel_dim if mel_dim is not None else 80,
                    upsample_initial_channel,
                    7,
                    1,
                    padding="same",
                    device=device,
                    dtype=dtype,
                )
            )
        )
        self.layers.extend(stream_layers[stream_idx : stream_idx + chunk_size])  # noqa
        stream_idx += chunk_size

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        ups = ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            out_pad = u % 2 if add_ups_out_pad else 0
            ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2 + out_pad,
                        output_padding=out_pad,
                        device=device,
                        dtype=dtype,
                    )
                )
            )
        ups.apply(init_weights)
        self.layers.extend(ups)
        self.layers.extend(stream_layers[stream_idx : stream_idx + chunk_size])  # noqa
        stream_idx += chunk_size

        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.layers.append(
                    ResBlock(
                        ch,
                        k,
                        d,
                    ).to(device, dtype=dtype)
                )
        self.layers.extend(stream_layers[stream_idx:])

        conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        conv_post.apply(init_weights)
        self.layers.append(conv_post)
        for u, k in zip(upsample_rates, upsample_kernel_sizes):
            assert k == 2 * u, (k, u)

        mean = torch.zeros((mel_dim,), dtype=torch.float)
        scale = torch.zeros((mel_dim,), dtype=torch.float)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

        self.gcmvn_mean = torch.tensor(gcmvn_mean, device=device, dtype=dtype)
        self.gcmvn_std = torch.tensor(gcmvn_std, device=device, dtype=dtype)

    def reset_parameters(self) -> None:
        for i in range(self.pn_layers):
            init.xavier_uniform_(
                self.layers[i][0].weight,
                init.calculate_gain("tanh" if i < self.pn_layers - 1 else "linear"),
            )

    def gcmvn_denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.gcmvn_mean is None or self.gcmvn_std is None:
            raise ValueError("gcmvn_mean is not set")

        assert (
            x.ndim == 3
            and x.shape[2] == self.gcmvn_mean.shape[0]
            and x.shape[2] == self.gcmvn_std.shape[0]
        )
        gcmvn_mean = self.gcmvn_mean.to(x)
        gcmvn_std = self.gcmvn_std.to(x)
        x = x * gcmvn_std.view(1, 1, -1).expand_as(x)  # type: ignore[attr-defined]
        return x + gcmvn_mean.view(1, 1, -1).expand_as(x)  # type: ignore[attr-defined,no-any-return]

    def forward(
        self,
        seqs: torch.Tensor,
        tgt_lang: str,
        prosody_input_seqs: torch.Tensor,
        padding_mask: Optional[PaddingMask] = None,
        prosody_padding_mask: Optional[PaddingMask] = None,
        durations: Optional[torch.Tensor] = None,
        duration_factor: float = 1.0,
        min_duration: int = 0,
        normalize_before: bool = True,
    ) -> List[torch.Tensor]:
        # Here we are adding batch dimension for the pretssel
        if seqs.ndim < 2:
            seqs = seqs.unsqueeze(0)
        if prosody_input_seqs.ndim < 3:
            prosody_input_seqs = prosody_input_seqs.unsqueeze(0)
        seqs, cond_embs = self.encoder_frontend(
            seqs,
            padding_mask,
            prosody_input_seqs,
            prosody_padding_mask,
            tgt_lang,
        )
        seqs, padding_mask = self.encoder(seqs, padding_mask, cond_embs)
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, durations, duration_factor, min_duration, cond_embs
        )
        seqs, padding_mask = self.decoder(seqs, padding_mask, cond_embs)
        seqs = self.final_proj(seqs)

        pn = seqs.transpose(1, 2)  # B x T x C -> B x C x T
        for i in range(self.pn_layers):
            pn = self.layers[i](pn)
        pn = pn.transpose(1, 2)

        x = seqs + pn
        x = self.gcmvn_denormalize(x)

        wavs = []
        for idx, _x in enumerate(x):
            _x = _x[: durations[idx].sum()]  # type: ignore[index]
            if normalize_before:
                _x = (_x - self.mean) / self.scale

            _x = _x.transpose(1, 0).unsqueeze(0)
            chunk_size = self.n_streams // 4
            _x = self.layers[self.pn_layers + chunk_size](_x)
            for i in range(self.num_upsamples):
                _x = F.leaky_relu(_x, LRELU_SLOPE)
                _x = self.layers[i + self.pn_layers + 1 + 2 * chunk_size](_x)
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.layers[
                            i * self.num_kernels
                            + j
                            + self.pn_layers
                            + 3 * chunk_size
                            + self.num_upsamples
                            + 1
                        ](_x)
                    else:
                        xs += self.layers[
                            i * self.num_kernels
                            + j
                            + self.pn_layers
                            + 3 * chunk_size
                            + self.num_upsamples
                            + 1
                        ](_x)
                _x = xs / self.num_kernels  # type: ignore
            _x = F.leaky_relu(_x)
            _x = self.layers[
                self.pn_layers
                + self.n_streams
                + self.num_upsamples * (1 + self.num_kernels)
                + 1
            ](_x)
            skip_output = _x
            h = skip_output

            for i1 in range(self.pn_layers, self.pn_layers + chunk_size):
                h = self.layers[i1](h)
            i1 += 2
            for i2 in range(i1, i1 + chunk_size):
                h = self.layers[i2](h)
            i2 = i2 + self.num_upsamples + 1

            for i3 in range(i2, i2 + chunk_size):
                h = self.layers[i3](h)
            i3 = i3 + (self.num_upsamples * self.num_kernels) + 1
            for i4 in range(i3, i3 + chunk_size):
                h = self.layers[i4](h)
            h = h[:, :, : _x.size(-1)]

            wavs.append(0.8 * h + torch.tanh(skip_output).squeeze(0))
        return wavs

    def remove_weight_norm(self) -> None:
        i = self.pn_layers + 1
        for j in range(self.num_upsamples):
            remove_weight_norm(self.layers[i + j])
        for k in range(self.num_upsamples * self.num_kernels):
            self.layers[i + j + k + 1].remove_weight_norm()
        remove_weight_norm(self.layers[self.pn_layers])
        remove_weight_norm(
            self.layers[
                self.pn_layers + 1 + self.num_upsamples * (1 + self.num_kernels)
            ]
        )
