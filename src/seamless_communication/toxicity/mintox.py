# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import logging
from typing import List, Optional, Tuple

from torch import Tensor
import torch
from torch.nn import functional as F


from seamless_communication.inference import SequenceGeneratorOptions
from seamless_communication.toxicity.etox_bad_word_checker import (
    ETOXBadWordChecker,
)
from fairseq2.generation import BannedSequenceProcessor
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.data.typing import StringLike
from fairseq2.typing import Device
from fairseq2.data import SequenceData
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYModel,
)


logger = logging.getLogger(__name__)


def _extract_bad_words_with_batch_indices(
    source_texts: List[StringLike],
    target_texts: List[StringLike],
    source_lang: str,
    target_lang: str,
    bad_word_checker: ETOXBadWordChecker,
) -> Tuple[List[str], List[int]]:
    all_bad_words, batch_indices = [], []

    for idx, (source_text, target_text) in enumerate(zip(source_texts, target_texts)):
        bad_words = bad_word_checker.extract_bad_words(
            str(source_text), str(target_text), source_lang, target_lang
        )

        if bad_words:
            batch_indices.append(idx)

            all_bad_words.extend(bad_words)

    return all_bad_words, batch_indices


def _replace_with_new_text_output_in_batch(
    original_texts: List[StringLike],
    indices_with_toxicity: List[int],
    new_texts: List[StringLike],
) -> None:
    new_idx = 0
    # indices_with_toxicity is a small list, using list should be fast enough.
    for original_idx in range(len(original_texts)):
        if original_idx in indices_with_toxicity:
            original_texts[original_idx] = new_texts[new_idx]
            new_idx += 1


def _replace_with_new_unit_output_in_batch(
    unit_tokenizer: UnitTokenizer,
    original_units: Tensor,
    indices_with_toxicity_tensor: Tensor,
    new_units: Tensor,
) -> None:
    original_units_length = original_units.size(1)
    new_units_length = new_units.size(1)
    length_diff = abs(new_units_length - original_units_length)
    nb_pads = (0, length_diff)
    pad_idx = unit_tokenizer.vocab_info.pad_idx or 1
    if new_units_length > original_units_length:
        # pad on the original units
        original_units = F.pad(
            original_units, pad=nb_pads, mode="constant", value=pad_idx
        )
    else:
        # pad on the new units
        new_units = F.pad(
            new_units, pad=nb_pads, mode="constant", value=pad_idx
        )
    original_units[indices_with_toxicity_tensor] = new_units


def mintox_pipeline(
    model: UnitYModel,
    text_tokenizer: TextTokenizer,
    unit_tokenizer: UnitTokenizer,
    device: Device,
    src_lang: str,
    tgt_lang: str,
    model_input: SequenceData,
    input_modality: "Modality",
    output_modality: "Modality",
    src_texts: List[StringLike],
    original_texts: List[StringLike],
    original_units: Optional[Tensor] = None,
    unit_generation_ngram_filtering: bool = False,
    text_generation_opts: Optional[SequenceGeneratorOptions] = None,
    unit_generation_opts: Optional[SequenceGeneratorOptions] = None,
    bad_word_checker: ETOXBadWordChecker = None,
    duration_factor: float = 1.0,
    prosody_encoder_input: Optional[SequenceData] = None,
) -> Tuple[List[StringLike], Optional[Tensor]]:
    """MinTox: Mitigation at INference time of added TOXicity."""
    from seamless_communication.inference.translator import Modality, Translator

    if text_generation_opts is None:
        text_generation_opts = SequenceGeneratorOptions(
            beam_size=5, soft_max_seq_len=(1, 200)
        )
    if unit_generation_opts is None:
        unit_generation_opts = SequenceGeneratorOptions(
            beam_size=5, soft_max_seq_len=(25, 50)
        )

    def _get_banned_sequence_processor(
        banned_sequences: List[str],
    ) -> BannedSequenceProcessor:
        text_encoder = text_tokenizer.create_raw_encoder(device=device)

        banned_seqs = [text_encoder(b) for b in banned_sequences]
        # A bannded string often appears after some puncatuations or symbols, we want
        # to include this sequence of token ids as well.
        # So we can ban not only the string "shit" but also "*shit", ",shit" etc.
        banned_seqs += [text_encoder(f"★{x}")[1:] for x in banned_sequences]
        return BannedSequenceProcessor(banned_seqs)

    bad_words, indices_with_toxicity = _extract_bad_words_with_batch_indices(
        src_texts,
        original_texts,
        src_lang,
        tgt_lang,
        bad_word_checker,
    )

    if len(indices_with_toxicity) == 0:
        # if no added toxicity is found, retrun the orignal output
        if output_modality == Modality.TEXT:
            return original_texts, None
        else:
            return original_texts, original_units
    else:
        logger.info(
            "TOX src_lang=%s tgt_lang=%s added_tox=%d",
            src_lang,
            tgt_lang,
            len(indices_with_toxicity),
        )
        # otherwise, redo the prediction with a list of bad words to ban
        banned_sequence_processor = _get_banned_sequence_processor(
            banned_sequences=list(set(bad_words)),
        )
        text_generation_opts.step_processor = banned_sequence_processor
        # select only the sources with toxicity
        indices_with_toxicity_tensor = torch.tensor(
            indices_with_toxicity, device=device
        )
        if model_input["is_ragged"]:
            model_input["seqs"] = torch.index_select(
                input=model_input["seqs"],
                dim=0,
                index=indices_with_toxicity_tensor,
            )
            model_input["seq_lens"] = torch.index_select(
                input=model_input["seq_lens"],
                dim=0,
                index=indices_with_toxicity_tensor,
            )
        seqs, padding_mask = get_seqs_and_padding_mask(model_input)
        # redo the prediction
        new_texts, new_units = Translator.get_prediction(
            model=model,
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            seqs=seqs,
            padding_mask=padding_mask,
            input_modality=input_modality,
            output_modality=output_modality,
            tgt_lang=tgt_lang,
            unit_generation_ngram_filtering=unit_generation_ngram_filtering,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            duration_factor=duration_factor,
            prosody_encoder_input=prosody_encoder_input,
        )
        batch_size = len(original_texts)
        if batch_size > 1:
            # reconstruct the text output by updating the original one in place
            _replace_with_new_text_output_in_batch(
                original_texts, indices_with_toxicity, new_texts
            )
            final_texts = original_texts
        else:
            final_texts = new_texts

        if output_modality == Modality.TEXT:
            return final_texts, None
        else:
            if batch_size > 1:
                assert original_units is not None
                assert new_units is not None
                # reconstruct the unit output by updating the original one in place
                _replace_with_new_unit_output_in_batch(
                    unit_tokenizer,
                    original_units,
                    indices_with_toxicity_tensor,
                    new_units,
                )
                final_units = original_units
            else:
                final_units = new_units
            return final_texts, final_units
