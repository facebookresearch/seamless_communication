# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, final

from fairseq2.models.encoder_decoder import EncoderDecoderModel, Seq2SeqDecoder
from fairseq2.models.seq2seq import Seq2SeqBatch
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2.nn.utils.module import check_model_dim
from overrides import final as finaloverride
from torch import Tensor
from torch.nn import Module


@final
class UnitYModel(EncoderDecoderModel):
    """Represents a UnitY model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`.

    Note that this implementation is augmented with a text encoder to enable
    translating from text.
    """

    model_dim: int
    input_modality: str
    speech_encoder_frontend: TransformerFrontend
    speech_encoder: TransformerEncoder
    text_encoder_frontend: Optional[TransformerFrontend]
    text_encoder: Optional[TransformerEncoder]
    text_decoder_frontend: TransformerFrontend
    text_decoder: TransformerDecoder
    final_proj: Projection
    t2u_model: Optional["UnitYT2UModel"]
    pad_idx: Optional[int]

    def __init__(
        self,
        speech_encoder_frontend: TransformerFrontend,
        speech_encoder: TransformerEncoder,
        text_encoder_frontend: Optional[TransformerFrontend],
        text_encoder: Optional[TransformerEncoder],
        text_decoder_frontend: TransformerFrontend,
        text_decoder: TransformerDecoder,
        final_proj: Projection,
        t2u_model: Optional["UnitYT2UModel"],
        pad_idx: Optional[int],
        input_modality: str = "speech",
    ) -> None:
        model_dim = speech_encoder.model_dim

        super().__init__(model_dim)

        self.input_modality = input_modality

        self.speech_encoder_frontend = speech_encoder_frontend
        self.speech_encoder = speech_encoder

        if text_encoder is not None:
            if text_encoder_frontend is None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_encoder_frontend` is `None`."
                )

            self.text_encoder_frontend = text_encoder_frontend
            self.text_encoder = text_encoder
        else:
            if text_encoder_frontend is not None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_encoder` is `None`."
                )

            self.register_module("text_encoder_frontend", None)
            self.register_module("text_encoder", None)

        self.text_decoder_frontend = text_decoder_frontend
        self.text_decoder = text_decoder

        self.final_proj = final_proj

        if t2u_model is not None:
            self.t2u_model = t2u_model
        else:
            self.register_module("t2u_model", None)

        self.pad_idx = pad_idx

        check_model_dim(self)

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.input_modality == "speech":
            return self.encode_speech(seqs, seq_lens)

        if self.input_modality == "text":
            return self.encode_text(seqs, seq_lens)

        raise RuntimeError(
            f"`input_modality` must be 'speech' or 'text', but is '{self.input_modality}' instead."
        )

    def encode_speech(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.speech_encoder_frontend(seqs, seq_lens)

        return self.speech_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    def encode_text(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.text_encoder is None:
            raise ValueError(
                "`encode_text()` requires a text encoder, but the current UnitY model does not have one."
            )

        assert self.text_encoder_frontend is not None

        seqs, padding_mask = self.text_encoder_frontend(seqs, seq_lens)

        return self.text_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.text_decoder_frontend(seqs, seq_lens, state_bag)

        return self.text_decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@final
class UnitYX2TModel(EncoderDecoderModel):
    model_dim: int
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    pad_idx: Optional[int]

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        pad_idx: Optional[int],
    ) -> None:
        model_dim = encoder.model_dim
        super().__init__(model_dim)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.pad_idx = pad_idx
        check_model_dim(self)

    @finaloverride
    def encode(
        self, seqs: Tensor, seq_lens: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.encoder_frontend(seqs, seq_lens)
        return self.encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag)

        return self.decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@final
class UnitYT2UModel(Module, Seq2SeqDecoder):
    """Represents a UnitY T2U model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`."""

    model_dim: int
    encoder: Optional[TransformerEncoder]
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection
    pad_idx: Optional[int]

    def __init__(
        self,
        encoder: Optional[TransformerEncoder],
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        pad_idx: Optional[int],
    ) -> None:
        super().__init__()

        self.model_dim = decoder.model_dim

        if encoder is not None:
            if encoder.model_dim != self.model_dim:
                raise ValueError(
                    f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {self.model_dim} instead."
                )

            self.encoder = encoder
        else:
            self.register_module("encoder", None)

        if decoder_frontend.model_dim != self.model_dim:
            raise ValueError(
                f"`model_dim` of `decoder_frontend` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.model_dim} and {self.model_dim} instead."
            )

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.pad_idx = pad_idx

    def forward(self, batch: Seq2SeqBatch) -> SequenceModelOutput:
        encoder_output, encoder_padding_mask = self.encode(
            batch.source_seqs, batch.source_seq_lens
        )

        decoder_output, decoder_padding_mask = self.decode(
            batch.target_seqs,
            batch.target_seq_lens,
            encoder_output,
            encoder_padding_mask,
        )

        return self.project(decoder_output, decoder_padding_mask)

    def encode(
        self,
        text_decoder_output: Tensor,
        text_decoder_padding_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.encoder is None:
            return text_decoder_output, text_decoder_padding_mask

        return self.encoder(text_decoder_output, text_decoder_padding_mask)  # type: ignore[no-any-return]

    def decode(
        self,
        seqs: Tensor,
        seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        seqs, padding_mask = self.decoder_frontend(seqs, seq_lens, state_bag)

        return self.decoder(  # type: ignore[no-any-return]
            seqs, padding_mask, encoder_output, encoder_padding_mask, state_bag
        )

    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[Tensor]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.pad_idx)


@dataclass
class UnitYOutput:
    """Holds the output of a UnitY model."""

    s2t_output: SequenceModelOutput
    """The S2T output of the multitask model."""

    mt_output: SequenceModelOutput
    """The MT output of the multitask model."""

    t2u_output: SequenceModelOutput
    """The output of the T2U model."""

    def compute_loss(
        self, targets: Tensor, ignore_prefix_size: int = 0, label_smoothing: float = 0.0
    ) -> None:
        # TODO: Implement R-Drop based loss
        pass
