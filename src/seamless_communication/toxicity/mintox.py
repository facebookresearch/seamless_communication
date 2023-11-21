# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple

from torch import Tensor
import torch
from torch.nn import functional as F


from seamless_communication.inference.generator import (
    SequenceToUnitOutput,
    SequenceGeneratorOptions,
)
from seamless_communication.toxicity.bad_word_checker import (
    BadWordChecker,
)
from fairseq2.generation import SequenceToTextOutput, BannedSequenceProcessor
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.data.typing import StringLike
from fairseq2.typing import Device
from fairseq2.data import SequenceData
from fairseq2.nn.padding import get_seqs_and_padding_mask
from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYModel,
)


def _extract_bad_words_with_batch_indices(
    source_texts: List[StringLike],
    target_texts: List[StringLike],
    source_lang: str,
    target_lang: str,
    bad_word_checker: BadWordChecker,
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
    original_text_out: SequenceToTextOutput,
    indices_with_toxicity: List[int],
    indices_with_toxicity_tensor: Tensor,
    new_text_output: SequenceToTextOutput,
    batch_size: int,
) -> None:
    original_text_out.encoder_output[
        indices_with_toxicity_tensor
    ] = new_text_output.encoder_output
    if original_text_out.encoder_padding_mask is not None:
        assert new_text_output.encoder_padding_mask is not None

        original_text_out.encoder_padding_mask.seq_lens[
            indices_with_toxicity_tensor
        ] = new_text_output.encoder_padding_mask.seq_lens

    new_i = 0
    for original_i in range(batch_size):
        if (
            original_i in indices_with_toxicity
        ):  # indices_with_toxicity is a small list, using list should be fast enough
            original_text_out.sentences[original_i] = new_text_output.sentences[new_i]
            original_text_out.generator_output.results[
                original_i
            ] = new_text_output.generator_output.results[new_i]
            new_i += 1


def _replace_with_new_unit_output_in_batch(
    unit_tokenizer: UnitTokenizer,
    original_unit_out: SequenceToUnitOutput,
    indices_with_toxicity: List[int],
    indices_with_toxicity_tensor: Tensor,
    new_unit_output: SequenceToUnitOutput,
    batch_size: int,
) -> None:
    original_units_length = original_unit_out.units.size(1)
    new_units_length = new_unit_output.units.size(1)
    length_diff = abs(new_units_length - original_units_length)
    nb_pads = (0, length_diff)
    pad_idx = unit_tokenizer.vocab_info.pad_idx or 1
    if new_units_length > original_units_length:
        # pad on the original units
        original_unit_out.units = F.pad(
            original_unit_out.units,
            pad=nb_pads,
            mode="constant",
            value=pad_idx,
        )
    else:
        # pad on the new units
        new_unit_output.units = F.pad(
            new_unit_output.units,
            pad=nb_pads,
            mode="constant",
            value=pad_idx,
        )
    original_unit_out.units[indices_with_toxicity_tensor] = new_unit_output.units

    new_i = 0
    if original_unit_out.generator_output is not None:
        for original_i in range(batch_size):
            if (
                original_i in indices_with_toxicity
                and new_unit_output.generator_output is not None
            ):
                original_unit_out.generator_output.results[
                    original_i
                ] = new_unit_output.generator_output.results[new_i]
                new_i += 1


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
    original_text_out: SequenceToTextOutput,
    original_unit_out: Optional[SequenceToUnitOutput] = None,
    unit_generation_ngram_filtering: bool = False,
    text_generation_opts: SequenceGeneratorOptions = SequenceGeneratorOptions(
        beam_size=5, soft_max_seq_len=(1, 200)
    ),
    unit_generation_opts: Optional[SequenceGeneratorOptions] = SequenceGeneratorOptions(
        beam_size=5, soft_max_seq_len=(25, 50)
    ),
    bad_word_checker: BadWordChecker = None,
) -> Tuple[SequenceToTextOutput, Optional[SequenceToUnitOutput]]:
    """MinTox: Mitigation at INference time of added TOXicity."""
    from seamless_communication.inference.translator import Modality, Translator

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
        original_text_out.sentences,
        src_lang,
        tgt_lang,
        bad_word_checker,
    )

    if len(indices_with_toxicity) == 0:
        # if no added toxicity is found, retrun the orignal output
        if output_modality == Modality.TEXT:
            return original_text_out, None
        else:
            return original_text_out, original_unit_out
    else:
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
        new_text_output, new_unit_output = Translator.get_prediction(
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
        )
        batch_size = len(original_text_out.sentences)
        if batch_size > 1:
            # reconstruct the text output by updating the original one in place
            _replace_with_new_text_output_in_batch(
                original_text_out,
                indices_with_toxicity,
                indices_with_toxicity_tensor,
                new_text_output,
                batch_size,
            )
            final_text_output = original_text_out
        else:
            final_text_output = new_text_output

        if output_modality == Modality.TEXT:
            return final_text_output, None
        else:
            if batch_size > 1:
                # reconstruct the unit output by updating the original one in place
                _replace_with_new_unit_output_in_batch(
                    unit_tokenizer,
                    original_unit_out,
                    indices_with_toxicity,
                    indices_with_toxicity_tensor,
                    new_unit_output,
                    batch_size,
                )
                final_unit_out = original_unit_out
            else:
                final_unit_out = new_unit_output
            return final_text_output, final_unit_out
