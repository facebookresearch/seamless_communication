from typing import Any, Optional, Sequence, Tuple, cast
from torch import Tensor
import torch
from fairseq2.device import Device
from fairseq2.data import Collater, SequenceData

def to_padding_mask(seq_lens: Tensor, batch_seq_len: int) -> Tensor:
    """Convert a sequence length array to a boolean padding mask tensor.

    :param seq_lens:
        An array where each element represents the length of a sequence. *Shape:*
        :math:`(N)`, where :math:`N` is the batch size.
    :param batch_seq_len:
        The sequence length of the mask.

    :returns:
        The mask. *Shape:* :math:`(N,S)`, where :math:`N` is the batch size and
        :math:`S` is the sequence length.
    """
    batch_size = seq_lens.size(0)

    # (N, S)
    indices = torch.arange(batch_seq_len, device=seq_lens.device).expand(batch_size, -1)

    # (N) -> (N, S)
    lengths = seq_lens.unsqueeze(1).expand(-1, batch_seq_len)

    return indices < lengths

def get_seqs_and_padding_mask(
    data,
) :
    """Return the sequences along with their padding mask from ``data``.

    :returns:
        - The sequences (i.e. `data["seqs"]`)
        - The padding mask of the returned sequences.
    """
    seqs = data["seqs"]

    if not data["is_ragged"]:
        return seqs, None

    return seqs, PaddingMask(data["seq_lens"], batch_seq_len=seqs.size(1))

class PaddingMask:
    """Represents a sequence padding mask."""

    seq_lens: Tensor
    batch_seq_len: int

    materialized: Optional[Tensor]

    def __init__(self, seq_lens: Tensor, batch_seq_len: int) -> None:
        """
        :param seq_lens:
            An array where each element represents the length of a sequence.
            *Shape:* :math:`(N)`, where :math:`N` is the batch size.
        :param batch_seq_len:
            The sequence length of the mask.
        """
        self.seq_lens = seq_lens
        self.batch_seq_len = batch_seq_len

        self.materialized = None

    def materialize(self) -> Tensor:
        """Materialize the boolean padding mask tensor."""
        if self.materialized is None:
            self.materialized = to_padding_mask(self.seq_lens, self.batch_seq_len)

        return self.materialized

    def trim(self, size: int) -> "PaddingMask":
        """Return a new trimmed padding mask.

        :param size:
            The amount by which to trim the sequences.
        """
        return PaddingMask(self.seq_lens - size, self.batch_seq_len - size)

    def to(self, device: Device) -> "PaddingMask":
        """Perform device conversion.

        :param device:
            The target device.
        """
        if self.seq_lens.device == device:
            return self

        return PaddingMask(self.seq_lens.to(device), self.batch_seq_len)

def apply_padding_mask(
    seqs: Tensor, padding_mask: Optional[PaddingMask], pad_value: Any = 0
) -> Tensor:
    """Apply the specified padding mask to ``seqs``.

    :param seqs:
        The sequences to mask. *Shape:* :math:`(N,S,*)`, where :math:`N` is the
        the batch size, :math:`S` is the sequence length, and :math:`*` is any
        number of sequence-specific dimensions including none.
    :param padding_mask:
        The padding mask to apply. *Shape:* :math:`(N,S)`, where :math:`N` is
        the batch size and :math:`S` is the sequence length.
    :param pad_value:
        The value for padded positions.

    :returns:
        The input sequences with mask applied. *Shape:* Same as ``seqs``.
    """
    if padding_mask is None:
        return seqs

    m = padding_mask.materialize()

    for _ in range(seqs.ndim - m.ndim):
        m = m.unsqueeze(-1)

    return seqs.where(m, pad_value)

def pad_seqs(
    seqs: Sequence[Tensor], pad_value: int = 0, pad_to_multiple: int = 1
) -> Tuple[Tensor, Optional[PaddingMask]]:
    """Stack ``seqs`` along a new batch dimension and pad them to equal length.

    :param seqs:
        The list of variable length sequences. All elements in ``seqs`` are
        expected to have the same shape except the first dimension.
    :param pad_value:
        The value for padded positions.
    :param pad_to_multiple:
        The sequence dimension is rounded up to the nearest multiple of the
        specified value.

    :returns:
        - The padded sequence stack. *Shape:* :math:`(N,S,*)`, where :math:`N`
          is the batch size, :math:`S` is the sequence length, and :math:`*` is
          any number of sequence-specific dimensions including none.
        - The padding mask of the sequence stack. *Shape:* :math:`(N,S)`, where
          :math:`N` is the batch size and :math:`S` is the sequence length.
    """
    collater = Collater(pad_value=pad_value, pad_to_multiple=pad_to_multiple)

    seq_data = cast(SequenceData, collater(seqs))

    return get_seqs_and_padding_mask(seq_data)