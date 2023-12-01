# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import codecs
import re
from pathlib import Path
from typing import Dict, List, Set, Union

from fairseq2.assets import (
    AssetCard,
    AssetDownloadManager,
    AssetStore,
    asset_store as base_asset_store,
    download_manager as base_download_manager,
)
from fairseq2.data import StringLike
from fairseq2.data.text import SentencePieceEncoder, SentencePieceModel


class ETOXBadWordChecker:
    bad_words: Dict[str, List[str]]
    bad_word_variants: Dict[str, Dict[str, List[str]]]
    sp_encoder: SentencePieceEncoder
    sp_langs: Set[str]

    def __init__(
        self,
        bad_words: Dict[str, List[str]],
        bad_word_variants: Dict[str, Dict[str, List[str]]],
        sp_encoder: SentencePieceEncoder,
        sp_langs: Set[str],
    ):
        self.bad_words = bad_words
        self.bad_word_variants = bad_word_variants
        self.sp_encoder = sp_encoder
        self.sp_langs = sp_langs

    def extract_bad_words(
        self,
        source_text: str,
        target_text: str,
        source_lang: str,
        target_lang: str,
    ) -> List[str]:
        bad_words_in_target_text = self.get_bad_words(
            target_text,
            target_lang,
        )

        # If there are no bad words in the target text, do nothing.
        if len(bad_words_in_target_text) == 0:
            return []

        bad_words_in_source_text = self.get_bad_words(
            source_text,
            source_lang,
        )

        # If there are bad words in the source text, do nothing.
        if len(bad_words_in_source_text) > 0:
            return []

        bad_words: List[str] = []

        for word in bad_words_in_target_text:
            bad_words.extend(self.bad_word_variants[target_lang][word])

        return bad_words

    def get_bad_words(self, text: str, lang: str) -> List[str]:
        try:
            bad_words = self.bad_words[lang]
        except KeyError as e:
            raise RuntimeError(f"MinTox model does not support {lang}.") from e

        text = self._preprocess(text)

        if lang in self.sp_langs:
            return self._find_bad_words_in_sp(text, bad_words)

        return self._find_bad_words(text, bad_words)

    @staticmethod
    def _preprocess(text: str) -> str:
        return re.sub(r"[\W+]", " ", text.lower())

    @staticmethod
    def _find_bad_words(text: str, bad_words: List[str]) -> List[str]:
        output: List[str] = []

        text = " " + text.lower() + " "

        bad_words = [" " + word.lower() + " " for word in bad_words]

        for word in bad_words:
            if word in text:
                output.append(word)

        return [word.strip(" ") for word in output]

    def _find_bad_words_in_sp(self, text: str, bad_words: List[str]) -> List[str]:
        text_tokens = self.sp_encoder.encode_as_tokens(text.lower())

        output: List[str] = []

        for word in bad_words:
            word_tokens = self.sp_encoder.encode_as_tokens(word.lower())

            if self._contains_tokens(text_tokens, word_tokens):
                output.append(str(word))

        return output

    @staticmethod
    def _contains_tokens(
        text_tokens: List[StringLike], word_tokens: List[StringLike]
    ) -> bool:
        for i in range(len(text_tokens) - len(word_tokens) + 1):
            for j in range(len(word_tokens)):
                if text_tokens[i + j] != word_tokens[j]:
                    break
            else:
                return True

        return False


class ETOXBadWordCheckerLoader:
    asset_store: AssetStore
    download_manager: AssetDownloadManager

    def __init__(
        self,
        asset_store: AssetStore,
        download_manager: AssetDownloadManager,
    ) -> None:
        self.asset_store = asset_store
        self.download_manager = download_manager

    def __call__(
        self,
        model_name_or_card: Union[str, AssetCard],
    ) -> ETOXBadWordChecker:
        if isinstance(model_name_or_card, AssetCard):
            card = model_name_or_card
        else:
            card = self.asset_store.retrieve_card(model_name_or_card)

        bad_words: Dict[str, List[str]] = {}

        bad_word_variants: Dict[str, Dict[str, List[str]]] = {}

        etox_lang_variants = card.field("etox_lang_variants").as_set(str)

        etox_ds_uri = card.field("etox_dataset").as_uri()

        etox_ds_path = self.download_manager.download_dataset(etox_ds_uri, "etox")

        for word_file in etox_ds_path.iterdir():
            lang = word_file.name[:8]

            if lang not in etox_lang_variants:
                lang = lang[:3]

            words = self._load_words(word_file)

            bad_words[lang] = words

            bad_word_variants[lang] = {}

            for word in words:
                bad_word_variants[lang][word] = [
                    word.lower(),
                    word.upper(),
                    word.capitalize(),
                ]

        sp_uri = card.field("sp_model").as_uri()

        sp_pathname = self.download_manager.download_tokenizer(sp_uri, card.name)

        sp_model = SentencePieceModel(sp_pathname)

        sp_encoder = SentencePieceEncoder(sp_model)

        sp_langs = card.field("sp_langs").as_set(str)

        return ETOXBadWordChecker(
            bad_words,
            bad_word_variants,
            sp_encoder,
            sp_langs,
        )

    @staticmethod
    def _load_words(pathname: Path) -> List[str]:
        words: List[str] = []

        with open(pathname, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                words.append(codecs.encode(line, "rot_13").rstrip("\n"))

        return list(set(words))  # Dedup.


load_etox_bad_word_checker = ETOXBadWordCheckerLoader(
    base_asset_store,
    base_download_manager,
)
