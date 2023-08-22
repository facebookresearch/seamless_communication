# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Sequence

import torch
from fairseq2.data import VocabularyInfo
from fairseq2.typing import Device
from torch import Tensor


class UnitTokenizer:
    """Represents a tokenizer to encode and decode UnitY speech units."""

    num_units: int
    langs: Sequence[str]
    lang_map: Dict[str, int]

    def __init__(self, num_units: int, langs: Sequence[str]) -> None:
        """
        :param num_units:
            The number of speech units.
        :param langs:
            The list of supported languages.
        """
        self.num_units = num_units

        self.langs = langs

        self.lang_map = {lang: idx for idx, lang in enumerate(langs)}

        # For legacy reasons, we have to repeat the language symbols twice,
        # along with a placeholder `<mask>` token.
        vocab_size = num_units + (2 * (len(langs) + 1)) + 4

        # We use fairseq's control symbol order.
        self.vocab_info = VocabularyInfo(
            size=vocab_size, bos_idx=0, pad_idx=1, eos_idx=2, unk_idx=3
        )

    def lang_to_index(self, lang: str) -> int:
        """Return the symbol index of the specified language."""
        # +4 for PAD/EOS/BOS/UNK, and +1 for the `<mask>` token.
        try:
            return self.num_units + len(self.langs) + self.lang_map[lang] + 5
        except KeyError:
            langs = ", ".join(self.langs)

            raise ValueError(
                f"`lang` must be one of the supported languages, but is '{lang}' instead. Supported languages: {langs}"
            )

    def index_to_lang(self, idx: int) -> str:
        """Return the language of the specified language symbol index."""
        relative_idx = idx - self.num_units - len(self.langs) - 5

        if relative_idx < 0 or relative_idx >= len(self.langs):
            raise ValueError(
                f"`idx` must correspond to one of the supported language symbol indices (0 to {len(self.langs) - 1}), but is {idx} instead."
            )

        return self.langs[relative_idx]

    def create_encoder(
        self, lang: str, device: Optional[Device] = None
    ) -> "UnitTokenEncoder":
        """Create a token encoder.

        :param lang:
            The language of generated token indices.
        """
        return UnitTokenEncoder(self, lang, device)

    def create_decoder(self) -> "UnitTokenDecoder":
        """Create a token decoder."""
        return UnitTokenDecoder(self)


class UnitTokenEncoder:
    """Encodes speech units into token indices."""

    tokenizer: UnitTokenizer
    eos_idx: int
    unk_idx: int
    lang_idx: int
    prefix_indices: Tensor

    def __init__(
        self, tokenizer: UnitTokenizer, lang: str, device: Optional[Device] = None
    ) -> None:
        """
        :param tokenizer:
            The unit tokenizer to use.
        :param lang:
            The language of generated token indices.
        """
        if not lang in tokenizer.lang_map:
            langs = ", ".join(tokenizer.langs)

            raise ValueError(
                f"`lang` must be one of the supported languages, but is '{lang}' instead. Supported languages: {langs}"
            )

        self.tokenizer = tokenizer

        assert tokenizer.vocab_info.eos_idx is not None
        assert tokenizer.vocab_info.unk_idx is not None

        self.eos_idx = tokenizer.vocab_info.eos_idx
        self.unk_idx = tokenizer.vocab_info.unk_idx

        self.lang_idx = tokenizer.lang_to_index(lang)

        if device is None:
            device = Device("cpu")

        # We always start sequences with EOS, followed by the language token.
        self.prefix_indices = torch.tensor(
            [self.eos_idx, self.lang_idx], device=device, dtype=torch.int64
        )

    def __call__(self, units: Tensor) -> Tensor:
        """Encode ``units`` to token indices.

        :param units:
            The speech units to encode. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            The token indices corresponding to ``units``. *Shape:*
            :math:`(N,S_{tok})` ,where :math:`N` is the batch size and
            :math`S_{tok}` is the sequence length of the token indices.
        """
        batch_size = units.size(0)

        token_indices = torch.cat(
            [self.prefix_indices.clone().expand(batch_size, -1), units.detach()], dim=1
        )

        # Ensure that non-symbol indices larger than `num_units` are replaced
        # with UNK.
        seqs = token_indices[:, 2:]

        # Add offset for control symbols.
        seqs += 4

        seqs[seqs >= self.tokenizer.num_units + 4] = self.unk_idx

        return token_indices


class UnitTokenDecoder:
    """Decodes speech units from token indices."""

    eos_idx: int
    pad_idx: int

    def __init__(self, tokenizer: UnitTokenizer) -> None:
        """
        :param tokenizer:
            The unit tokenizer to use.
        """
        assert tokenizer.vocab_info.eos_idx is not None
        assert tokenizer.vocab_info.pad_idx is not None

        self.eos_idx = tokenizer.vocab_info.eos_idx
        self.pad_idx = tokenizer.vocab_info.pad_idx

    def __call__(self, token_indices: Tensor) -> Tensor:
        """Decode ``token_indices`` to speech units.

        :param token_indices:
            The token indices to decode. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            The speech units corresponding to ``token_indices``. *Shape:*
            :math:`(N,S_{unt})`, where :math:`N` is the batch size and
            :math`S_{unt}` is the sequence length of the speech units.
        """
        if token_indices.size(1) == 0:
            return token_indices

        # Remove the prefix EOS symbol. The language symbol is still expected to
        # be part of the decoded output.
        units = token_indices[:, 1:].clone().detach()

        # Also, replace EOS with PAD at sequence ends.
        units[units == self.eos_idx] = self.pad_idx

        units[units == self.pad_idx] = self.pad_idx + 4

        # Remove offset of control symbols (exclude language symbol).
        units[:, 1:] -= 4

        return units
