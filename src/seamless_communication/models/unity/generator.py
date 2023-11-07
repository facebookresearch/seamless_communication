# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch

from torch import Tensor
from fairseq2.data import VocabularyInfo
from fairseq2.data.text import TextTokenizer
from fairseq2.generation import (
    Seq2SeqGenerator,
    SequenceGeneratorOptions,
    SequenceGeneratorOutput,
    SequenceToTextGenerator,
    SequenceToTextOutput,
)
from fairseq2.nn.padding import PaddingMask, apply_padding_mask
from fairseq2.nn.utils.module import infer_device

from seamless_communication.models.unity.model import (
    UnitYModel,
    UnitYX2TModel,
    UnitYT2UModel,
)
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenDecoder,
    UnitTokenizer,
)


def remove_consecutive_repeated_ngrams(
    sequence: List[int], min_size: int = 1, max_size: int = 40
):
    assert 1 <= min_size <= max_size
    drop_idx = set()  # indices that will be dropped from the sequence

    # start from the beginning, check if an ngram of size k (for k=max..min) is
    # followed by its copy, if so delete the first one, and start over after
    # the deleted ngram.
    start = 0
    while start < len(sequence):
        for k in range(max_size, min_size - 1, -1):
            if sequence[start : start + k] == sequence[start + k : start + k + k]:
                drop_idx |= set(range(start, start + k))
                start += k - 1  # assumes repeating subsequences don't overlap
                break
        start += 1
    return [token for idx, token in enumerate(sequence) if idx not in drop_idx]


class UnitYGenerator:
    """Generates text translations and speech units from a UnitY model."""

    model: UnitYModel
    s2t_generator: SequenceToTextGenerator
    t2t_generator: Optional[SequenceToTextGenerator]
    unit_decoder: Optional[UnitTokenDecoder]
    unit_generator: Optional[Seq2SeqGenerator]

    def __init__(
        self,
        model: UnitYModel,
        text_tokenizer: TextTokenizer,
        target_lang: str,
        unit_tokenizer: Optional[UnitTokenizer] = None,
        text_opts: Optional[SequenceGeneratorOptions] = None,
        unit_opts: Optional[SequenceGeneratorOptions] = None,
    ) -> None:
        """
        :param model:
            The UnitY model to use for generation.
        :param text_tokenizer:
            The text tokenizer to use.
        :param unit_tokenizer:
            The unit tokenizer to use.
        :param target_lang:
            The target language.
        :param text_generator_opts:
            The options to pass to the underlying text :class:`Seq2SeqGenerator`.
        :param unit_generator_opts:
            The options to pass to the underlying unit :class:`Seq2SeqGenerator`.
        """
        model.eval()

        self.model = model

        s2t_model = UnitYX2TModel(
            encoder_frontend=model.speech_encoder_frontend,
            encoder=model.speech_encoder,
            decoder_frontend=model.text_decoder_frontend,
            decoder=model.text_decoder,
            final_proj=model.final_proj,
            target_vocab_info=model.target_vocab_info,
        )
        self.s2t_generator = SequenceToTextGenerator(
            s2t_model, text_tokenizer, target_lang, text_opts
        )

        if model.text_encoder is None:
            self.t2t_generator = None
        else:
            assert model.text_encoder_frontend is not None
            assert model.text_encoder is not None
            t2t_model = UnitYX2TModel(
                encoder_frontend=model.text_encoder_frontend,
                encoder=model.text_encoder,
                decoder_frontend=model.text_decoder_frontend,
                decoder=model.text_decoder,
                final_proj=model.final_proj,
                target_vocab_info=model.target_vocab_info,
            )
            self.t2t_generator = SequenceToTextGenerator(
                t2t_model, text_tokenizer, target_lang, text_opts
            )

        self.unit_generator = None
        self.unit_decoder = None
        # Set up unit generator.
        if unit_tokenizer is not None:
            if model.t2u_model is None:
                raise ValueError(
                    "`model` does not have a T2U sub-model when `unit_tokenizer` is not None."
                )

            self.unit_decoder = unit_tokenizer.create_decoder()

            unit_encoder = unit_tokenizer.create_encoder(
                lang=target_lang, device=infer_device(model.t2u_model)
            )

            if isinstance(self.model.t2u_model, UnitYT2UModel):
                if unit_opts is None:
                    # Speech sequences are typically much longer than text sequences.
                    unit_opts = SequenceGeneratorOptions(
                        soft_max_seq_len=(1, 50), hard_max_seq_len=5000
                    )

                self.unit_generator = Seq2SeqGenerator(
                    self.model.t2u_model,
                    unit_tokenizer.vocab_info,
                    unit_encoder.prefix_indices,
                    unit_opts,
                )

    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
        input_modality: str = "speech",
        output_modality: str = "speech",
        ngram_filtering: bool = False,
    ) -> Tuple[SequenceToTextOutput, Optional["SequenceToUnitOutput"]]:
        """
        :param source_seqs:
            The source sequences to use for generation. *Shape:* :math:`(N,S,*)`,
            where :math:`N` is the batch size, :math:`S` is the sequence length,
            and :math:`*` is any number of sequence-specific dimensions
            including none.
        :param source_padding_mask:
            The padding mask of ``source_seqs``. *Shape:* :math:`(N,S)`, where
            :math:`N` is the batch size and :math:`S` is the sequence length.
        :param input_modality:
            The type of modality to encode.
        :param output_modality:
            The type of modality to decode.
        :param ngram_filtering:
            If True, removes consecutive repeated ngrams
            from the decoded unit output.

        :returns:
            - The output of the text generator.
            - The output of the unit generator.
        """

        if input_modality == "speech":
            text_output = self.s2t_generator.generate_ex(
                source_seqs, source_padding_mask
            )
        elif input_modality == "text" and self.t2t_generator is not None:
            text_output = self.t2t_generator.generate_ex(
                source_seqs, source_padding_mask
            )
        elif input_modality == "text" and self.t2t_generator is None:
            raise ValueError(
                f"Please set use_text_encoder to True in your model config to encode text."
            )
        else:
            raise ValueError(f"Unsupported input_modality: {input_modality}")

        # We skip T2U when we only need to output text.
        if output_modality == "text":
            return text_output, None

        text_seqs, text_padding_mask = text_output.generator_output.collate()

        # Manually trim the final EOS token to be consistent with fairseq.
        text_seqs = text_seqs[:, :-1]

        if text_padding_mask is not None:
            text_padding_mask = text_padding_mask.trim(1)

        # Use the output of the text generator to compute the decoder output.
        decoder_output, decoder_padding_mask = self.model.decode(
            text_seqs,
            text_padding_mask,
            text_output.encoder_output,
            text_output.encoder_padding_mask,
        )

        assert self.model.t2u_model is not None
        assert self.unit_decoder is not None

        unit_gen_output = None
        if isinstance(self.model.t2u_model, UnitYT2UModel):
            assert self.unit_generator is not None
            t2u_encoder_output, t2u_encoder_padding_mask = self.model.t2u_model.encode(
                decoder_output, decoder_padding_mask
            )
            unit_gen_output = self.unit_generator(
                t2u_encoder_output,
                t2u_encoder_padding_mask,
                source_seq_len=source_seqs.size(1),
            )
            unit_seqs, _ = unit_gen_output.collate()
        else:
            unit_decoder_output, decoder_padding_mask = self.model.t2u_model(
                text_decoder_output=decoder_output,
                text_decoder_padding_mask=decoder_padding_mask,
                text_seqs=text_seqs,
            )
            # (B, S_unit, V_unit)
            unit_seqs = unit_decoder_output.logits.argmax(dim=2)
            # Apply the padding mask to the generated units.
            unit_seqs = apply_padding_mask(
                unit_seqs, decoder_padding_mask, unit_decoder_output.vocab_info.pad_idx
            )

        # Convert to speech units.
        units = self.unit_decoder(unit_seqs)

        if ngram_filtering:
            units = remove_consecutive_repeated_ngrams(units.cpu().numpy().tolist())
            units = torch.tensor(units)

        unit_output = SequenceToUnitOutput(units, unit_gen_output)

        return text_output, unit_output


@dataclass
class SequenceToUnitOutput:
    units: Tensor
    """The generated units."""

    generator_output: Optional[SequenceGeneratorOutput]
    """The output of the underlying :class:`Seq2SeqGenerator`."""
