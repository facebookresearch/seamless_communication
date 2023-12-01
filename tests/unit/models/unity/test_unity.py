# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import pytest
import torch

from seamless_communication.models.unity import UnitTokenizer
from tests.common import assert_equal, device


class TestUnitTokenizer:
    def test_init_works(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        assert tokenizer.num_units == 100

        assert tokenizer.lang_map == {"eng": 0, "deu": 1, "fra": 2}

        assert tokenizer.vocab_info.size == 112

    def test_lang_to_index_works(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        assert tokenizer.lang_to_index("eng") == 108
        assert tokenizer.lang_to_index("deu") == 109
        assert tokenizer.lang_to_index("fra") == 110

    def test_lang_to_index_works_nar_decoder(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100,
            langs=["eng", "deu", "fra"],
            model_arch="seamlessM4T_large_v2",
        )
        assert tokenizer.vocab_info.size == 108

        assert tokenizer.lang_to_index("eng") == 104
        assert tokenizer.lang_to_index("deu") == 105
        assert tokenizer.lang_to_index("fra") == 106

    def test_lang_to_index_raises_error_when_lang_is_not_supported(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        with pytest.raises(
            ValueError,
            match=r"^`lang` must be one of the supported languages, but is 'foo' instead\. Supported languages: eng, deu, fra$",
        ):
            tokenizer.lang_to_index("foo")

    def test_index_to_lang_works(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        assert tokenizer.index_to_lang(108) == "eng"
        assert tokenizer.index_to_lang(109) == "deu"
        assert tokenizer.index_to_lang(110) == "fra"

    def test_index_to_lang_works_nar_decoder(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100,
            langs=["eng", "deu", "fra"],
            model_arch="seamlessM4T_large_v2",
        )

        assert tokenizer.index_to_lang(104) == "eng"
        assert tokenizer.index_to_lang(105) == "deu"
        assert tokenizer.index_to_lang(106) == "fra"

    def test_vocab_control_symbols(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        assert tokenizer.vocab_info.bos_idx == 0
        assert tokenizer.vocab_info.pad_idx == 1
        assert tokenizer.vocab_info.eos_idx == 2
        assert tokenizer.vocab_info.unk_idx == 3

    def test_index_to_lang_raises_error_when_idx_is_out_of_range(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        with pytest.raises(
            ValueError,
            match=r"^`idx` must correspond to one of the supported language symbol indices \(0 to 2\), but is 1234 instead\.$",
        ):
            tokenizer.index_to_lang(1234)


class TestUnitEncoder:
    def test_init_raises_error_when_lang_is_not_supported(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        with pytest.raises(
            ValueError,
            match=r"^`lang` must be one of the supported languages\, but is 'xyz' instead\. Supported languages: eng, deu, fra$",
        ):
            tokenizer.create_encoder(lang="xyz", device=device)

    def test_call_works(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        prefix = torch.tensor([2, 109], device=device, dtype=torch.int64)

        encoder = tokenizer.create_encoder(lang="deu", device=device)

        # Empty units.
        units = torch.ones((1, 0), device=device, dtype=torch.int64)

        assert_equal(encoder(units), prefix.expand(1, -1))

        # Batched units.
        units = torch.ones((6, 4), device=device, dtype=torch.int64)

        assert_equal(
            encoder(units), torch.cat([prefix.expand(6, -1), units + 4], dim=1)
        )

    def test_call_works_nar_decoder(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100,
            langs=["eng", "deu", "fra"],
            model_arch="seamlessM4T_large_v2",
        )

        encoder = tokenizer.create_encoder(lang="deu", device=device)

        # Empty units.
        units = torch.ones((1, 0), device=device, dtype=torch.int64)

        assert_equal(encoder(units), units)

        # Batched units.
        units = torch.ones((6, 4), device=device, dtype=torch.int64)

        assert_equal(encoder(units), units + 4)

    def test_call_works_when_units_have_unks(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        encoder = tokenizer.create_encoder(lang="deu", device=device)

        units = torch.ones((6, 4), device=device, dtype=torch.int64)

        units[1, 3] = 100
        units[2, 1] = 101

        token_indices = encoder(units)

        assert token_indices[1, 5].item() == tokenizer.vocab_info.unk_idx
        assert token_indices[2, 3].item() == tokenizer.vocab_info.unk_idx

    def test_call_works_when_units_have_unks_nar_decoder(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100,
            langs=["eng", "deu", "fra"],
            model_arch="seamlessM4T_large_v2",
        )

        encoder = tokenizer.create_encoder(lang="deu", device=device)

        units = torch.ones((6, 4), device=device, dtype=torch.int64)

        units[1, 3] = 100
        units[2, 1] = 101

        token_indices = encoder(units)

        assert token_indices[1, 3].item() == tokenizer.vocab_info.unk_idx
        assert token_indices[2, 1].item() == tokenizer.vocab_info.unk_idx


class TestUnitDecoder:
    def test_call_works(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100, langs=["eng", "deu", "fra"], model_arch="seamlessM4T_large"
        )

        encoder = tokenizer.create_encoder(lang="deu", device=device)
        decoder = tokenizer.create_decoder()

        assert tokenizer.vocab_info.eos_idx is not None
        assert tokenizer.vocab_info.pad_idx is not None

        units1 = torch.ones((6, 4), device=device, dtype=torch.int64)

        encoded_units = encoder(units1)

        encoded_units[2, 2] = tokenizer.vocab_info.eos_idx

        units2 = decoder(encoded_units)

        units1[2, 2] = tokenizer.vocab_info.pad_idx

        prefix = torch.tensor([109], device=device, dtype=torch.int64)

        assert_equal(torch.cat([prefix.expand(6, -1), units1], dim=1), units2)

    def test_call_works_nar_decoder(self) -> None:
        tokenizer = UnitTokenizer(
            num_units=100,
            langs=["eng", "deu", "fra"],
            model_arch="seamlessM4T_large_v2",
        )

        encoder = tokenizer.create_encoder(lang="deu", device=device)
        decoder = tokenizer.create_decoder()

        assert tokenizer.vocab_info.eos_idx is not None
        assert tokenizer.vocab_info.pad_idx is not None

        units1 = torch.ones((6, 4), device=device, dtype=torch.int64)

        encoded_units = encoder(units1)

        encoded_units[2, 2] = tokenizer.vocab_info.eos_idx

        units2 = decoder(encoded_units)

        units1[2, 2] = tokenizer.vocab_info.pad_idx

        assert_equal(units1, units2)
