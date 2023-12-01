# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from fairseq2.nn.padding import PaddingMask, to_padding_mask
from torch import Tensor
from torch.nn import Conv1d, LayerNorm, Module, ModuleList, ReLU, Sigmoid, Tanh, init


class ECAPA_TDNN(Module):
    """
    Represents the ECAPA-TDNN model described in paper:
    :cite:t`https://doi.org/10.48550/arxiv.2005.07143`.

    Arguments
    ---------
    :param channels:
        Output channels for TDNN/SERes2Net layer.
    :param kernel_sizes:
        List of kernel sizes for each layer.
    :param dilations:
        List of dilations for kernels in each layer.
    :param groups:
        List of groups for kernels in each layer.
    """

    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        dilations: List[int],
        attention_channels: int,
        res2net_scale: int,
        se_channels: int,
        global_context: bool,
        groups: List[int],
        embed_dim: int,
        input_dim: int,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(dilations)
        self.channels = channels
        self.embed_dim = embed_dim
        self.blocks = ModuleList()

        self.blocks.append(
            TDNNBlock(
                input_dim,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                groups[0],
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    groups=groups[i],
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            groups=groups[-1],
        )

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_norm = LayerNorm(channels[-1] * 2, eps=1e-12)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters and buffers of the module."""

        def encoder_init(m: Module) -> None:
            if isinstance(m, Conv1d):
                init.xavier_uniform_(m.weight, init.calculate_gain("relu"))

        self.apply(encoder_init)

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[PaddingMask] = None,
    ) -> Tensor:
        """Returns the embedding vector.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        x = x.transpose(1, 2)

        xl = []
        for layer in self.blocks:
            x = layer(x, padding_mask=padding_mask)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # Attentive Statistical Pooling
        x = self.asp(x, padding_mask=padding_mask)
        x = self.asp_norm(x.transpose(1, 2)).transpose(1, 2)

        # Final linear transformation
        x = self.fc(x)

        x = x.transpose(1, 2).squeeze(1)  # B x C
        return F.normalize(x, dim=-1)


class TDNNBlock(Module):
    """An implementation of TDNN.

    Arguments
    ----------
    :param in_channels : int
        Number of input channels.
    :param out_channels : int
        The number of output channels.
    :param kernel_size : int
        The kernel size of the TDNN blocks.
    :param dilation : int
        The dilation of the TDNN block.
    :param groups: int
        The groups size of the TDNN blocks.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = TDNNBlock(64, 64, kernel_size=3, dilation=1)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            groups=groups,
        )
        self.activation = ReLU()
        self.norm = LayerNorm(out_channels, eps=1e-12)

    def forward(self, x: Tensor, padding_mask: Optional[PaddingMask] = None) -> Tensor:
        """Processes the input tensor x and returns an output tensor."""
        x = self.activation(self.conv(x))

        return self.norm(x.transpose(1, 2)).transpose(1, 2)  # type: ignore[no-any-return]


class Res2NetBlock(Module):
    """An implementation of Res2NetBlock w/ dilation.

    Arguments
    ---------
    :param in_channels : int
        The number of channels expected in the input.
    :param out_channels : int
        The number of output channels.
    :param scale : int
        The scale of the Res2Net block.
    :param kernel_size: int
        The kernel size of the Res2Net block.
    :param dilation : int
        The dilation of the Res2Net block.

    Example
    -------
    >>> inp_tensor = torch.rand([8, 120, 64]).transpose(1, 2)
    >>> layer = Res2NetBlock(64, 64, scale=4, dilation=3)
    >>> out_tensor = layer(inp_tensor).transpose(1, 2)
    >>> out_tensor.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for i in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """Processes the input tensor x and returns an output tensor."""
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)

        y_tensor = torch.cat(y, dim=1)
        return y_tensor


class SEBlock(Module):
    """An implementation of squeeze-and-excitation block.

    Arguments
    ---------
    in_channels : int
        The number of input channels.
    se_channels : int
        The number of output channels after squeeze.
    out_channels : int
        The number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        se_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.conv1 = Conv1d(
            in_channels=in_channels, out_channels=se_channels, kernel_size=1
        )
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv1d(
            in_channels=se_channels, out_channels=out_channels, kernel_size=1
        )
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, padding_mask: Optional[PaddingMask] = None) -> Tensor:
        """Processes the input tensor x and returns an output tensor."""
        if padding_mask is not None:
            mask = padding_mask.materialize().unsqueeze(1)
            s = (x * mask).sum(dim=2, keepdim=True) / padding_mask.seq_lens[
                :, None, None
            ]
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))

        return s * x


class AttentiveStatisticsPooling(Module):
    """This class implements an attentive statistic pooling layer for each channel.
    It returns the concatenated mean and std of the input tensor.

    Arguments
    ---------
    channels: int
        The number of input channels.
    attention_channels: int
        The number of attention channels.
    """

    def __init__(
        self, channels: int, attention_channels: int = 128, global_context: bool = True
    ):
        super().__init__()

        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)

        self.tanh = Tanh()
        self.conv = Conv1d(
            in_channels=attention_channels, out_channels=channels, kernel_size=1
        )

    def forward(self, x: Tensor, padding_mask: Optional[PaddingMask] = None) -> Tensor:
        """Calculates mean and std for a batch (input tensor).

        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape [N, C, L].
        """
        L = x.shape[-1]

        def _compute_statistics(
            x: Tensor, m: Tensor, dim: int = 2, eps: float = self.eps
        ) -> Tuple[Tensor, Tensor]:
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std

        # if lengths is None:
        #     lengths = [x.shape[0]]

        # Make binary mask of shape [N, 1, L]
        # mask = to_padding_mask(lengths, max(lengths))
        if padding_mask is not None:
            mask = padding_mask.materialize()
        else:
            mask = to_padding_mask(torch.IntTensor([L]), L).repeat(x.shape[0], 1).to(x)
        mask = mask.unsqueeze(1)

        # Expand the temporal context of the pooling layer by allowing the
        # self-attention to look at global properties of the utterance.
        if self.global_context:
            # torch.std is unstable for backward computation
            # https://github.com/pytorch/pytorch/issues/4320
            total = mask.sum(dim=2, keepdim=True).to(x)
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        # Apply layers
        attn = self.conv(self.tanh(self.tdnn(attn)))

        # Filter out zero-paddings
        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)

        return pooled_stats


class SERes2NetBlock(Module):
    """An implementation of building block in ECAPA-TDNN, i.e.,
    TDNN-Res2Net-TDNN-SEBlock.

    Arguments
    ----------
    out_channels: int
        The number of output channels.
    res2net_scale: int
        The scale of the Res2Net block.
    kernel_size: int
        The kernel size of the TDNN blocks.
    dilation: int
        The dilation of the Res2Net block.
    groups: int
    Number of blocked connections from input channels to output channels.

    Example
    -------
    >>> x = torch.rand(8, 120, 64).transpose(1, 2)
    >>> conv = SERes2NetBlock(64, 64, res2net_scale=4)
    >>> out = conv(x).transpose(1, 2)
    >>> out.shape
    torch.Size([8, 120, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels,
            out_channels,
            res2net_scale,
            kernel_size,
            dilation,
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x: Tensor, padding_mask: Optional[PaddingMask] = None) -> Tensor:
        """Processes the input tensor x and returns an output tensor."""
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, padding_mask=padding_mask)

        return x + residual
