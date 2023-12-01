# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from fairseq2.data import SequenceData, StringLike
from fairseq2.data.text import TextTokenizer
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    Seq2SeqGenerator,
    SequenceToTextConverter,
    StepProcessor,
)
from fairseq2.nn.padding import (
    PaddingMask,
    apply_padding_mask,
    get_seqs_and_padding_mask,
    pad_seqs,
)
from fairseq2.nn.utils.module import infer_device
from torch import Tensor

from seamless_communication.models.unity.model import (
    UnitYModel,
    UnitYT2UModel,
    UnitYX2TModel,
)
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenDecoder,
    UnitTokenizer,
)


def remove_consecutive_repeated_ngrams(
    sequence: List[int], min_size: int = 1, max_size: int = 40
) -> List[int]:
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


@dataclass
class SequenceGeneratorOptions:
    """Holds the options to pass to a sequence generator."""

    beam_size: int = 5
    """The beam size."""

    soft_max_seq_len: Tuple[int, int] = (1, 200)
    """The terms ``a`` and ``b`` of ``ax + b`` where ``x`` is the source
    sequence length. The generated sequences (including prefix sequence) will
    have the maximum length of ``min(hard_max_seq_len, ax + b)``. See also
    ``hard_max_seq_len``."""

    hard_max_seq_len: int = 1024
    """The hard limit on maximum length of generated sequences."""

    step_processor: Optional[StepProcessor] = None
    """The processor called at each generation step."""

    unk_penalty: float = 0.0
    """The UNK symbol penalty, where values less than 0 produce more UNKs;
    values greater than 0 produce fewer UNKs."""

    len_penalty: float = 1.0
    """The length penalty, where values less than 1.0 favor shorter
    sequences; values greater than 1.0 favor longer sequences."""


class UnitYGenerator:
    """Generates text translations and speech units from a UnitY model."""

    model: UnitYModel
    s2t_converter: SequenceToTextConverter
    t2t_converter: Optional[SequenceToTextConverter]
    unit_decoder: Optional[UnitTokenDecoder]
    unit_prefix_indices: Optional[Tensor]
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

        if text_opts is None:
            text_opts = SequenceGeneratorOptions()

        if model.text_decoder is None:
            raise ValueError(
                "`UnitYGenerator` requires a text decoder, but the current UnitY model does not have one."
            )
        assert model.text_decoder_frontend is not None
        assert model.final_proj is not None

        s2t_model = UnitYX2TModel(
            encoder_frontend=model.speech_encoder_frontend,
            encoder=model.speech_encoder,
            decoder_frontend=model.text_decoder_frontend,
            decoder=model.text_decoder,
            final_proj=model.final_proj,
            target_vocab_info=model.target_vocab_info,
        )

        step_processors = []
        if text_opts.step_processor is not None:
            step_processors.append(text_opts.step_processor)

        generator = BeamSearchSeq2SeqGenerator(
            s2t_model,
            beam_size=text_opts.beam_size,
            max_gen_len=text_opts.soft_max_seq_len,
            max_seq_len=text_opts.hard_max_seq_len,
            echo_prompt=True,
            step_processors=step_processors,
            unk_penalty=text_opts.unk_penalty,
            len_penalty=text_opts.len_penalty,
        )
        self.s2t_converter = SequenceToTextConverter(
            generator, text_tokenizer, "translation", target_lang
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
            generator = BeamSearchSeq2SeqGenerator(
                t2t_model,
                beam_size=text_opts.beam_size,
                max_gen_len=text_opts.soft_max_seq_len,
                max_seq_len=text_opts.hard_max_seq_len,
                echo_prompt=True,
                step_processors=step_processors,
                unk_penalty=text_opts.unk_penalty,
                len_penalty=text_opts.len_penalty,
            )
            self.t2t_converter = SequenceToTextConverter(
                generator, text_tokenizer, "translation", target_lang
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

            self.unit_prefix_indices = unit_encoder.prefix_indices

            if isinstance(self.model.t2u_model, UnitYT2UModel):
                if unit_opts is None:
                    # Speech sequences are typically much longer than text sequences.
                    unit_opts = SequenceGeneratorOptions(
                        soft_max_seq_len=(25, 50), hard_max_seq_len=5000
                    )

                step_processors = []
                if unit_opts.step_processor is not None:
                    step_processors.append(unit_opts.step_processor)

                self.unit_generator = BeamSearchSeq2SeqGenerator(
                    self.model.t2u_model,
                    beam_size=unit_opts.beam_size,
                    max_gen_len=unit_opts.soft_max_seq_len,
                    max_seq_len=unit_opts.hard_max_seq_len,
                    echo_prompt=True,
                    step_processors=step_processors,
                    unk_penalty=unit_opts.unk_penalty,
                    len_penalty=unit_opts.len_penalty,
                )

    @torch.inference_mode()
    def __call__(
        self,
        source_seqs: Tensor,
        source_padding_mask: Optional[PaddingMask],
        input_modality: str = "speech",
        output_modality: str = "speech",
        ngram_filtering: bool = False,
        duration_factor: float = 1.0,
        prosody_encoder_input: Optional[SequenceData] = None,
    ) -> Tuple[List[StringLike], Optional[Tensor]]:
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
            texts, text_gen_output = self.s2t_converter.batch_convert(
                source_seqs, source_padding_mask
            )
        elif input_modality == "text":
            if self.t2t_converter is None:
                raise ValueError(
                    "Please set `use_text_encoder` to `True` in your model config to encode text."
                )
            texts, text_gen_output = self.t2t_converter.batch_convert(
                source_seqs, source_padding_mask
            )
        else:
            raise ValueError(f"Unsupported input_modality: {input_modality}")

        # We skip T2U when we only need to output text.
        if output_modality == "text":
            return texts, None

        assert self.model.target_vocab_info.pad_idx is not None

        text_seq_list = [h[0].seq for h in text_gen_output.hypotheses]

        text_seqs, text_padding_mask = pad_seqs(
            text_seq_list, self.model.target_vocab_info.pad_idx
        )

        # Manually trim the final EOS token to be consistent with fairseq.
        text_seqs = text_seqs[:, :-1]

        if text_padding_mask is not None:
            text_padding_mask = text_padding_mask.trim(1)

        # Use the output of the text generator to compute the decoder output.
        decoder_output, decoder_padding_mask = self.model.decode(
            text_seqs,
            text_padding_mask,
            text_gen_output.encoder_output,
            text_gen_output.encoder_padding_mask,
        )

        assert self.model.t2u_model is not None
        assert self.unit_decoder is not None

        unit_gen_output = None
        prosody_encoder_out = None
        if self.model.prosody_encoder_model is not None:
            assert prosody_encoder_input is not None
            prosody_input_seqs, prosody_padding_mask = get_seqs_and_padding_mask(
                prosody_encoder_input
            )
            prosody_encoder_out = self.model.prosody_encoder_model(
                prosody_input_seqs,
                prosody_padding_mask,
            ).unsqueeze(1)

        if isinstance(self.model.t2u_model, UnitYT2UModel):
            assert self.unit_generator is not None
            assert self.unit_prefix_indices is not None

            # (S_pre) -> (N, S_pre)
            prefix_seqs = self.unit_prefix_indices.expand(decoder_output.size(0), -1)

            unit_gen_output = self.unit_generator(
                source_seqs=decoder_output,
                source_padding_mask=decoder_padding_mask,
                prompt_seqs=prefix_seqs,
                prompt_padding_mask=None,
            )

            assert self.model.t2u_model.target_vocab_info.pad_idx is not None

            unit_seq_list = [h[0].seq for h in unit_gen_output.hypotheses]

            unit_seqs, _ = pad_seqs(
                unit_seq_list, self.model.t2u_model.target_vocab_info.pad_idx
            )
        else:
            t2u_model_output, decoder_padding_mask, _ = self.model.t2u_model(
                text_decoder_output=decoder_output,
                text_decoder_padding_mask=decoder_padding_mask,
                text_seqs=text_seqs,
                duration_factor=duration_factor,
                film_cond_emb=prosody_encoder_out,
            )
            # (B, S_unit, V_unit)
            unit_seqs = t2u_model_output.logits.argmax(dim=2)
            # Apply the padding mask to the generated units.
            unit_seqs = apply_padding_mask(
                unit_seqs, decoder_padding_mask, t2u_model_output.vocab_info.pad_idx
            )

        # Convert to speech units.
        units = self.unit_decoder(unit_seqs)

        # ngram-filtering doesn't apply to NAR unit decoding.
        if ngram_filtering and isinstance(self.model.t2u_model, UnitYT2UModel):
            if units.size(0) > 1:
                raise NotImplementedError(
                    "unit ngram_filtering is not implemented for batch_size > 1."
                )
            arr = remove_consecutive_repeated_ngrams(units[0].tolist())
            units = torch.tensor(arr).to(units).unsqueeze(0)

        return texts, units
