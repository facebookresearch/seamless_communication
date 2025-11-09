
from abc import ABC, abstractmethod
from typing import Optional, Protocol, final

from torch import Tensor
import torch
from fairseq2.data_type import DataType
from fairseq2.device import Device
from overrides import final
finaloverride = final
from fairseq2.nn.incremental_state import IncrementalStateBag

def _create_causal_attention_mask(
    seq_len: int,
    key_len: int,
    attn_window_len: Optional[int],
    device: Optional[Device],
    dtype: Optional[DataType],
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    # As of PyTorch 2.0, `triu` does not support bf16.
    dt = torch.float32 if dtype == torch.bfloat16 else dtype

    mask = torch.ones((seq_len, key_len), device=device, dtype=dt)

    mask.tril_(diagonal=0)

    if attn_window_len is not None:
        mask.triu_(diagonal=1 - attn_window_len)

    mask.log_()

    return mask.to(dtype)

class AttentionMask(ABC):
    """Represents an attention mask."""

    materialized: Optional[Tensor]

    def __init__(self) -> None:
        self.materialized = None

    def materialize(self) -> Tensor:
        """Materialize the attention mask tensor."""
        if self.materialized is None:
            self.materialized = self._do_materialize()

        return self.materialized

    @abstractmethod
    def _do_materialize(self) -> Tensor:
        ...


class AttentionMaskFactory(Protocol):
    """Constructs instances of :class:`AttentionMask`."""

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[AttentionMask]:
        """
        :param seqs:
            The sequences for which to create a mask. *Shape:* :math:`(N,S,M)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`M` is the dimensionality of the model.
        :param keys:
            The keys. *Shape:* :math:`(N,S_{kv},K)`, where :math:`N` is the
            batch size, :math:`S_{kv}` is the key/value sequence length, and
            :math:`K` is the key size.
        :param training:
            If ``True``, indicates that the calling module is in training mode.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            An implementation-defined mask for ``seqs``.
        """


@final
class CustomAttentionMask(AttentionMask):
    """Represents a custom attention mask provided by the user."""

    def __init__(self, mask: Tensor) -> None:
        """
        :param mask:
            The custom attention mask tensor.
        """
        super().__init__()

        self.mask = mask

    @finaloverride
    def _do_materialize(self) -> Tensor:
        return self.mask


@final
class CausalAttentionMask(AttentionMask):
    """Represents a causal attention mask.

    *Shape:* :math:`(S,S_{kv})`, where :math:`S` is the sequence length and
    :math:`S_{kv}` is the key/value sequence length.

    Usage:

    >>> import torch
    >>>
    >>> from fairseq2.nn.transformer import CausalAttentionMask
    >>>
    >>> mask = CausalAttentionMask(seq_len=4, key_len=6)
    >>> mask.materialize()
    tensor([[0., -inf, -inf, -inf, -inf, -inf],
            [0.,   0., -inf, -inf, -inf, -inf],
            [0.,   0.,   0., -inf, -inf, -inf],
            [0.,   0.,   0.,   0., -inf, -inf]])
    >>>
    >>> mask = CausalAttentionMask(seq_len=4, key_len=4, attn_window_len=2)
    >>> mask.materialize()
    tensor([[0.,   -inf, -inf, -inf],
            [0.,     0., -inf, -inf],
            [-inf,   0.,   0., -inf],
            [-inf, -inf,   0.,   0.]])
    """

    def __init__(
        self,
        seq_len: int,
        key_len: int,
        *,
        attn_window_len: Optional[int] = None,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ) -> None:
        """
        :param seq_len:
            The sequence length.
        :param key_len:
            The key/value sequence length.
        :param attn_window_len:
            The attention window length as described in Section 3.1 of
            :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`. If ``None``,
            constructs a full causal attention mask.
        """
        super().__init__()

        self.seq_len = seq_len
        self.key_len = key_len
        self.attn_window_len = attn_window_len

        self.device, self.dtype = device, dtype

    @finaloverride
    def _do_materialize(self) -> Tensor:
        return _create_causal_attention_mask(
            self.seq_len, self.key_len, self.attn_window_len, self.device, self.dtype
        )


class CausalAttentionMaskFactory:
    """Constructs instances of :class:`CausalAttentionMask`."""

    def __init__(self, *, attn_window_len: Optional[int] = None) -> None:
        """
        :param attn_window_len:
            The attention window length as described in Section 3.1 of
            :cite:t:`https://doi.org/10.48550/arxiv.2004.05150`. If ``None``,
            constructs a full causal attention mask.
        """
        self.attn_window_len = attn_window_len

    def __call__(
        self,
        seqs: Tensor,
        keys: Tensor,
        *,
        training: bool = True,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Optional[CausalAttentionMask]:
        seq_len, key_len = seqs.size(1), keys.size(1)

        if seq_len > key_len:
            raise ValueError(
                f"The sequence length of `seqs` must be less than or equal to the sequence length of `keys` ({key_len}), but is {seq_len} instead."
            )

        if seq_len <= 1:
            # Return `None` if the sequence has a length of 1 during training;
            # or if we attend to past steps during incremental decoding.
            return None

        return CausalAttentionMask(
            seq_len,
            key_len,
            attn_window_len=self.attn_window_len,
            device=seqs.device,
            dtype=seqs.dtype,
        )

    def __repr__(self) -> str:
        if self.attn_window_len is None:
            return "CausalAttentionMaskFactory()"

        return f"CausalAttentionMaskFactory(attn_window_len={self.attn_window_len})"

