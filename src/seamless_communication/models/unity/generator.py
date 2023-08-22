# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from fairseq2.data.text import TextTokenizer
from fairseq2.generation import (
    Seq2SeqGenerator,
    SequenceGeneratorOptions,
    SequenceGeneratorOutput,
    SequenceToTextGenerator,
    SequenceToTextOutput,
)
from seamless_communication.models.unity.model import UnitYModel, UnitYX2TModel
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenDecoder,
    UnitTokenizer,
)
from fairseq2.nn.utils.module import infer_device
from torch import Tensor


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
        if model.t2u_model is None:
            raise ValueError(
                "`model` does not have a T2U sub-model. "
                "For text generation only, "
                "use `SequenceToTextGenerator` instead."
            )

        model.eval()

        self.model = model

        s2t_model = UnitYX2TModel(
            encoder_frontend=model.speech_encoder_frontend,
            encoder=model.speech_encoder,
            decoder_frontend=model.text_decoder_frontend,
            decoder=model.text_decoder,
            final_proj=model.final_proj,
            pad_idx=model.pad_idx,
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
                pad_idx=model.pad_idx,
            )
            self.t2t_generator = SequenceToTextGenerator(
                t2t_model, text_tokenizer, target_lang, text_opts
            )

        self.unit_generator = None
        self.unit_decoder = None
        # Set up unit generator.
        if unit_tokenizer is not None:
            self.unit_decoder = unit_tokenizer.create_decoder()

            unit_encoder = unit_tokenizer.create_encoder(
                lang=target_lang, device=infer_device(model.t2u_model)
            )

            if unit_opts is None:
                # Speech sequences are typically much longer than text sequences.
                unit_opts = SequenceGeneratorOptions(
                    soft_max_seq_len=(1, 50), hard_max_seq_len=5000
                )

            self.unit_generator = Seq2SeqGenerator(
                model.t2u_model,
                unit_tokenizer.vocab_info,
                unit_encoder.prefix_indices,
                unit_opts,
            )

    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_seq_lens: Optional[Tensor],
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
        :param source_seq_lens:
            An array where each element represents the length of the sequence at
            the same index in ``source_seqs``. *Shape:* :math:`(N)`, where
            :math:`N` is the batch size.
        :param input_modality:
            The type of modality to encode.
        :param output_modality:
            The type of modality to decode.

        :returns:
            - The output of the text generator.
            - The output of the unit generator.
        """

        if input_modality == "speech":
            text_output = self.s2t_generator.generate_ex(source_seqs, source_seq_lens)
        elif input_modality == "text" and self.t2t_generator is not None:
            text_output = self.t2t_generator.generate_ex(source_seqs, source_seq_lens)
        elif input_modality == "text" and self.t2t_generator is None:
            raise ValueError(
                f"Please set use_text_encoder to True in your model config to encode text."
            )
        else:
            raise ValueError(f"Unsupported input_modality: {input_modality}")

        # We skip T2U when we only need to output text.
        if output_modality == "text":
            return text_output, None

        text_seqs, text_seq_lens = text_output.generator_output.collate()

        # Use the output of the text generator to compute the decoder output.
        decoder_output, decoder_padding_mask = self.model.decode(
            text_seqs,
            text_seq_lens,
            text_output.encoder_output,
            text_output.encoder_padding_mask,
        )

        assert self.model.t2u_model is not None

        t2u_encoder_output, t2u_encoder_padding_mask = self.model.t2u_model.encode(
            decoder_output, decoder_padding_mask
        )

        assert self.unit_generator is not None
        assert self.unit_decoder is not None

        unit_gen_output = self.unit_generator(
            t2u_encoder_output,
            t2u_encoder_padding_mask,
            source_seq_len=source_seqs.size(1),
        )

        unit_seqs, _ = unit_gen_output.collate()

        # Convert to speech units.
        units = self.unit_decoder(unit_seqs)
        if ngram_filtering:
            units = remove_consecutive_repeated_ngrams(units.cpu().numpy().tolist())
            units = torch.tensor(units)

        unit_output = SequenceToUnitOutput(
            units, unit_gen_output, t2u_encoder_output, t2u_encoder_padding_mask
        )

        return text_output, unit_output


@dataclass
class SequenceToUnitOutput:
    units: Tensor
    """The generated units."""

    generator_output: SequenceGeneratorOutput
    """The output of the underlying :class:`Seq2SeqGenerator`."""

    t2u_encoder_output: Tensor
    """The encoder output of the underlying UnitY T2U model used to generate the
    units. *Shape:* :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
    :math:`S_{enc}` is the encoder output sequence length, and :math:`M` is the
    dimensionality of the model."""

    t2u_encoder_padding_mask: Optional[Tensor]
    """The float padding mask of :attr:`encoder_output`. *Shape:*
    :math:`(N,S_{enc})`, where :math:`N` is the batch size and :math:`S_{enc}`
    is the encoder output sequence length."""
