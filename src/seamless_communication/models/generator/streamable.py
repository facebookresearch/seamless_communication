# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import math
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeVar

import torch
from fairseq2.typing import DataType, Device
from torch.nn import (
    ELU,
    LSTM,
    Conv1d,
    ConvTranspose1d,
    GroupNorm,
    Identity,
    Module,
    Sequential,
)
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm  # type: ignore[attr-defined]

CONV_NORMALIZATIONS = frozenset(
    ["none", "weight_norm", "spectral_norm", "time_group_norm"]
)


def apply_parametrization_norm(
    module: Module,
    norm: Literal["none", "weight_norm", "spectral_norm", "time_group_norm"] = "none",
) -> Module:
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(  # type: ignore[no-untyped-def]
    module: Module,
    causal: bool = False,
    norm: Literal["none", "weight_norm", "spectral_norm", "time_group_norm"] = "none",
    **norm_kwargs,
) -> Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, torch.nn.modules.conv._ConvNd)
        return GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> torch.Tensor:
    """Pad for a convolution to make sure that the last window is full.
    Extra padding is added at the end. This is required to ensure that we can rebuild
    an output of the same length, as otherwise, even with padding, some time steps
    might get removed.
    For instance, with total padding = 4, kernel size = 4, stride = 2:
        0 0 1 2 3 4 5 0 0   # (0s are padding)
        1   2   3           # (output frames of a convolution, last 0 is never used)
        0 0 1 2 3 4 5 0     # (output of tr. conv., but pos. 5 is going to get removed as padding)
            1 2 3 4         # once you removed padding, we are missing one time step !
    """
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))  # noqa


def pad1d(
    x: torch.Tensor,
    paddings: Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
) -> torch.Tensor:
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: Tuple[int, int]) -> torch.Tensor:
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


class NormConv1d(Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: Literal[
            "none", "weight_norm", "spectral_norm", "time_group_norm"
        ] = "none",
        norm_kwargs: Dict[str, Any] = {},
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.conv: Module = apply_parametrization_norm(
            Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                dilation=dilation,
                groups=groups,
                bias=bias,
                device=device,
                dtype=dtype,
            ),
            norm,
        )
        self.norm: Module = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: Literal[
            "none", "weight_norm", "spectral_norm", "time_group_norm"
        ] = "none",
        norm_kwargs: Dict[str, Any] = {},
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.convtr = apply_parametrization_norm(
            ConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                device=device,
                dtype=dtype,
            ),
            norm,
        )
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convtr(x)
        x = self.norm(x)
        return x


class StreamableConv1d(Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: Literal[
            "none", "weight_norm", "spectral_norm", "time_group_norm"
        ] = "none",
        norm_kwargs: Dict[str, Any] = {},
        pad_mode: str = "reflect",
        activation: Optional[Module] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.activation = activation
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            device=device,
            dtype=dtype,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation:
            x = self.activation(x)
        kernel_size: int = self.conv.conv.kernel_size[0]  # type: ignore[index,assignment]
        stride: int = self.conv.conv.stride[0]  # type: ignore[index,assignment]
        dilation = self.conv.conv.dilation[0]  # type: ignore[index]
        kernel_size = (  # type: ignore[assignment]
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)  # type: ignore[no-any-return]


class StreamableConvTranspose1d(Module):
    """ConvTranspose1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        norm: Literal[
            "none", "weight_norm", "spectral_norm", "time_group_norm"
        ] = "none",
        trim_right_ratio: float = 1.0,
        norm_kwargs: Dict[str, Any] = {},
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.convtr = NormConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
            device=device,
            dtype=dtype,
        )
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert (
            self.causal or self.trim_right_ratio == 1.0
        ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0.0 and self.trim_right_ratio <= 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel_size: int = self.convtr.convtr.kernel_size[0]  # type: ignore[index,assignment]
        stride: int = self.convtr.convtr.stride[0]  # type: ignore[index,assignment]
        padding_total = kernel_size - stride

        y: torch.Tensor = self.convtr(x)

        # We will only trim fixed padding. Extra padding from `pad_for_conv1d` would be
        # removed at the very end, when keeping only the right length for the output,
        # as removing it here would require also passing the length at the matching layer
        # in the encoder.
        if self.causal:
            # Trim the padding on the right according to the specified ratio
            # if trim_right_ratio = 1.0, trim everything from right
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y


class StreamableLSTM(Module):
    """LSTM without worrying about the hidden state, nor the layout of the data.
    Expects input as convolutional layout.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        skip: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        self.skip = skip
        self.lstm = LSTM(dimension, dimension, num_layers, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        y = y.permute(1, 2, 0)
        return y  # type: ignore[no-any-return]


class StreamableResnetBlock(Module):
    """custom Residual block model with streamable convnet.

    Args:
        dim (int): Dimension of the input/output.
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation_params (dict): Parameters to provide to the (ELU) activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: List[int] = [3, 1],
        dilations: List[int] = [1, 1],
        activation_params: Dict[str, Any] = {"alpha": 1.0},
        norm: Literal[
            "none", "weight_norm", "spectral_norm", "time_group_norm"
        ] = "none",
        norm_params: Dict[str, Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                ELU(**activation_params),
                StreamableConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                    device=device,
                    dtype=dtype,
                ),
            ]
        self.block = Sequential(*block)
        self.shortcut: Module
        if true_skip:
            self.shortcut = Identity()
        else:
            self.shortcut = StreamableConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
                device=device,
                dtype=dtype,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.block(x)  # type: ignore[no-any-return]
