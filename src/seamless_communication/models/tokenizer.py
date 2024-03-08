# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Sequence, Set, final

from fairseq2.data.text import (
    SentencePieceTokenizer,
    SentencePieceEncoder,
)
from fairseq2.typing import Device, override


@final
class SPMTokenizer(SentencePieceTokenizer):
    """Represents standard SPM-based tokenizer used in MT tasks"""

    langs: Set[str]
    prepend_target_langtok_to_target: bool

    def __init__(
        self,
        path: Path,
        langs: Sequence[str],
        prepend_target_langtok_to_target: bool = True,
    ) -> None:
        """
        :param pathname:
            The pathname of the SentencePiece model file.
        :param langs:
            The list of supported languages.
        :param default_lang:
            The fall-back language if no language is specified.
        """
        # Each language is represented by a `__lang__` control symbol.
        control_symbols = [self._lang_tok_to_internal(lang) for lang in sorted(langs)]

        super().__init__(path, control_symbols)

        self.langs = set(langs)
        self.prepend_target_langtok_to_target = prepend_target_langtok_to_target

    @classmethod
    def _lang_tok_to_internal(cls, lang: str) -> str:
        return f"__{lang}__"

    @override
    def create_encoder(
        self,
        *,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        mode: Optional[str] = None,
        device: Optional[Device] = None,
        pin_memory: bool = False,
    ) -> SentencePieceEncoder:
        """Create a token encoder.

        :param task:
            Must be 'translation'. If ``None``, defaults to 'translation'.
        :param lang:
            A language from :attr:`langs`. If ``None``, defaults to
            :attr:`default_lang`.
        :param mode:
            Must be 'source' or 'target'.
        :param device:
            The device on which to construct tensors.
        :param pin_memory:
            If ``True``, uses pinned memory while constructing tensors.
        """
        if task is not None and task != "translation":
            raise ValueError(f"`task` must be 'translation', but is '{task}' instead.")

        assert lang is not None

        if lang not in self.langs:
            raise ValueError(
                f"`lang` must be a supported language, but is '{lang}' instead."
            )

        if mode is None or mode == "source":
            prefix_tokens = []
            suffix_tokens = ["</s>"]
        elif mode == "target":
            prefix_tokens = (
                ["</s>"] + [self._lang_tok_to_internal(lang)]
                if self.prepend_target_langtok_to_target
                else []
            )
            suffix_tokens = ["</s>"]
        else:
            raise ValueError(
                f"`mode` must be 'source' or 'target', but is '{mode}' instead."
            )

        return SentencePieceEncoder(
            self.model,
            prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens,
            device=device,
            pin_memory=pin_memory,
        )
