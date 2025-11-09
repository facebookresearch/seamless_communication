# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union, final
from typing_extensions import override

from fairseq2.data.tokenizers import VocabularyInfo
# from fairseq2.models.encoder_decoder import EncoderDecoderModel
# from fairseq2.models.sequence import SequenceModelOutput
# from fairseq2.models.seq2seq import Seq2SeqModel
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.projection import Projection
from fairseq2.models.transformer import TransformerDecoder, TransformerEncoder
from overrides import final as finaloverride
from seamless_communication.padding import PaddingMask
from torch import Tensor
import torch
from torch.nn import Module
from torch.nn.functional import log_softmax, nll_loss

from fairseq2.data.tokenizers import VocabularyInfo
from seamless_communication.models.generator.ecapa_tdnn import ECAPA_TDNN
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.nar_decoder_frontend import NARDecoderFrontend

@dataclass
class SequenceModelOutput:
    """Holds the output of a sequence model."""

    logits: Tensor
    """The logits for next-step prediction. *Shape:* :math:`(N,S,T)`, where
    :math:`N` is the batch size, :math:`S` is the sequence length, and :math:`T`
    is the size of the vocabulary."""

    vocab_size: int
    """The vocabulary information."""

    pad_idx: int

    def compute_loss(
        self,
        targets: Tensor,
        *,
        ignore_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """Compute the negative log-likelihood loss.

        :param targets:
            The target indices. *Shape:* :math:`(N,S)`, where :math:`N` is the
            batch size and :math:`S` is the sequence length.
        :param ignore_prefix_size:
            The number of steps from the beginning of the sequence that should
            be ignored in the loss computation.
        :param label_smoothing:
            The amount of label smoothing to apply while computing the loss.
        """
        if ignore_prefix_size > 0:
            logits = self.logits[:, ignore_prefix_size:, :]
        else:
            logits = self.logits

        if ignore_prefix_size > 0:
            targets = targets[:, ignore_prefix_size:]

        # For numerical stability run in single precision.
        lprobs = log_softmax(logits, dim=-1, dtype=torch.float32)

        return nll_loss(
            lprobs, targets, self.pad_idx, label_smoothing=label_smoothing
        )

class EncoderDecoderModel(Module):
    """Represents an encoder-decoder model."""

    model_dim: int

    def __init__(self, model_dim: int, target_vocab_size: int) -> None:
        """
        :param model_dim:
            The dimensionality of the model.
        :param target_vocab_info:
            The vocabulary information of sequences produced by the model.
        """
        super().__init__()
        self.target_vocab_size=target_vocab_size
        self.model_dim = model_dim

    @override
    def forward(self, batch) -> SequenceModelOutput:
        encoder_output, encoder_padding_mask = self.encode(
            batch.source_seqs, batch.source_padding_mask
        )

        decoder_output, decoder_padding_mask = self.decode(
            batch.target_seqs,
            batch.target_padding_mask,
            encoder_output,
            encoder_padding_mask,
        )

        return self.project(decoder_output, decoder_padding_mask)

    @abstractmethod
    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Encode the specified source sequences.

        :param seqs:
            The source sequences to encode. *Shape:* :math:`(N,S_{src},*)`,
            where :math:`N` is the batch size, :math:`S_{src}` is the source
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.

        :returns:
            - The encoder output. *Shape:* :math:`(N,S_{out},M)`, where
              :math:`N` is the batch size, :math:`S_{out}` is the output
              sequence length, and :math:`M` is the dimensionality of the model.
            - The padding mask of the encoder output. *Shape:*
              :math:`(N,S_{out})`, where :math:`N` is the batch size and
              :math:`S_{out}` is the output sequence length.
        """

    @abstractmethod
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        """Decode the specified target sequences.

        :param seqs:
            The target sequences to decode. *Shape:* :math:`(N,S_{tgt},*)`,
            where :math:`N` is the batch size, :math:`S_{tgt}` is the target
            sequence length, and :math:`*` is any number of sequence-specific
            dimensions including none.
        :param padding_mask:
            The padding mask of ``seqs``. *Shape:* :math:`(N,S)`, where :math:`N`
            is the batch size and :math:`S` is the sequence length.
        :param encoder_output:
            The encoder output to use in encoder-decoder attention. *Shape:*
            :math:`(N,S_{enc},M)`, where :math:`N` is the batch size,
            :math:`S_{enc}` is the encoder output sequence length, and :math:`M`
            is the dimensionality of the model.
        :param encoder_padding_mask:
            The padding mask of ``encoder_output``. *Shape:* :math:`(N,S_{enc})`,
            where :math:`N` is the batch size and :math:`S_{enc}` is the encoder
            output sequence length.
        :param state_bag:
            The state bag to use for incremental decoding.

        :returns:
            - The decoder output. *Shape:* :math:`(N,S_{tgt},M)`, where
              :math:`N` is the batch size, :math:`S_{tgt}` is the target
              sequence length, and :math:`M` is the dimensionality of the model.
            - The padding mask of the decoder output. *Shape:*
              :math:`(N,S_{tgt})`, where :math:`N` is the batch size and
              :math:`S_{tgt}` is the target sequence length.
        """

    @abstractmethod
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        """Produce logits for next-step prediction.

        :param decoder_output:
            The decoder output. *Shape:* :math:`(N,S_{tgt},M)`, where :math:`N`
            is the batch size, :math:`S_{tgt}` is the target sequence length,
            and :math:`M` is the dimensionality of the model.
        :param decoder_padding_mask:
            The padding mask of the decoder output. *Shape:* :math:`(N,S_{tgt})`,
            where :math:`N` is the batch size and :math:`S_{tgt}` is the target
            sequence length.
        """

    def extra_repr(self) -> str:
        """:meta private:"""
        return f"model_dim={self.model_dim}"
    
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
    text_decoder_frontend: Optional[TransformerFrontend]
    text_decoder: Optional[TransformerDecoder]
    final_proj: Optional[Projection]
    t2u_model: Union["UnitYT2UModel", "UnitYNART2UModel", None]
    prosody_encoder_model: Optional[ECAPA_TDNN]

    def __init__(
        self,
        speech_encoder_frontend: TransformerFrontend,
        speech_encoder: TransformerEncoder,
        text_encoder_frontend: Optional[TransformerFrontend],
        text_encoder: Optional[TransformerEncoder],
        text_decoder_frontend: Optional[TransformerFrontend],
        text_decoder: Optional[TransformerDecoder],
        final_proj: Optional[Projection],
        t2u_model: Union["UnitYT2UModel", "UnitYNART2UModel", None],
        target_vocab_size: int = None,
        pad_idx: int = None,
        prosody_encoder_model: Optional[ECAPA_TDNN] = None,
        model_dim: int = None,
        input_modality: str = "speech",
    ) -> None:
        # model_dim = speech_encoder.model_dim

        super().__init__(model_dim, target_vocab_size)
        self.pad_idx=pad_idx

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

            # self.register_module("text_encoder_frontend", None)
            # self.register_module("text_encoder", None)

        if text_decoder is not None:
            if text_decoder_frontend is None:
                raise ValueError(
                    "Both `text_decoder` and `text_decoder_frontend` must be specified, but `text_decoder_frontend` is `None`."
                )

            self.text_decoder_frontend = text_decoder_frontend
            self.text_decoder = text_decoder
            self.final_proj = final_proj
        else:
            if text_decoder_frontend is not None:
                raise ValueError(
                    "Both `text_encoder` and `text_encoder_frontend` must be specified, but `text_decoder` is `None`."
                )

            # self.register_module("text_decoder_frontend", None)
            # self.register_module("text_decoder", None)
            # self.register_module("final_proj", None)

        if t2u_model is not None:
            self.t2u_model = t2u_model
        # else:
        #     self.register_module("t2u_model", None)

        self.target_vocab_size = target_vocab_size
        if prosody_encoder_model is not None:
            self.prosody_encoder_model = prosody_encoder_model
        # else:
        #     self.register_module("prosody_encoder_model", None)

    @finaloverride
    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.input_modality == "speech":
            return self.encode_speech(seqs, padding_mask)

        if self.input_modality == "text":
            return self.encode_text(seqs, padding_mask)

        raise RuntimeError(
            f"`input_modality` must be 'speech' or 'text', but is '{self.input_modality}' instead."
        )

    def encode_speech(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.speech_encoder_frontend(seqs, padding_mask)

        return self.speech_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    def encode_text(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.text_encoder is None:
            raise ValueError(
                "`encode_text()` requires a text encoder, but the current UnitY model does not have one."
            )

        assert self.text_encoder_frontend is not None

        seqs, padding_mask = self.text_encoder_frontend(seqs, padding_mask)

        return self.text_encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.text_decoder is None:
            raise ValueError(
                "`decode()` requires a text decoder, but the current UnitY model does not have one."
            )

        assert self.text_decoder_frontend is not None

        seqs, padding_mask = self.text_decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.text_decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        if self.final_proj is None:
            raise ValueError(
                "`project()` requires a final_proj layer, but the current UnitY model does not have one."
            )

        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_size, self.pad_idx)


@final
class UnitYX2TModel(EncoderDecoderModel):
    model_dim: int
    encoder_frontend: TransformerFrontend
    encoder: TransformerEncoder
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        encoder_frontend: TransformerFrontend,
        encoder: TransformerEncoder,
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_vocab_info: VocabularyInfo,
    ) -> None:
        model_dim = encoder.model_dim

        super().__init__(model_dim, target_vocab_info)

        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.decoder_frontend = decoder_frontend
        self.decoder = decoder
        self.final_proj = final_proj
        self.target_vocab_info = target_vocab_info

    @finaloverride
    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.encoder_frontend(seqs, padding_mask)
        return self.encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @finaloverride
    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    @finaloverride
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_info)


@final
class UnitYT2UModel(EncoderDecoderModel):
    """Represents a UnitY T2U model as described in
    :cite:t`https://doi.org/10.48550/arxiv.2212.08055`."""

    encoder: Optional[TransformerEncoder]
    decoder_frontend: TransformerFrontend
    decoder: TransformerDecoder
    final_proj: Projection

    def __init__(
        self,
        encoder: Optional[TransformerEncoder],
        decoder_frontend: TransformerFrontend,
        decoder: TransformerDecoder,
        final_proj: Projection,
        target_vocab_info: VocabularyInfo,
    ) -> None:
        super().__init__(decoder.model_dim, target_vocab_info)

        if encoder is not None:
            self.encoder = encoder
        else:
            self.register_module("encoder", None)

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.encoder is None:
            return seqs, padding_mask

        return self.encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    def decode(
        self,
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        *,
        state_bag: Optional[IncrementalStateBag] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.decoder_frontend(
            seqs, padding_mask, state_bag=state_bag
        )

        return self.decoder(  # type: ignore[no-any-return]
            seqs,
            padding_mask,
            encoder_output,
            encoder_padding_mask,
            state_bag=state_bag,
        )

    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_info)


@final
class UnitYNART2UModel(Module):
    """Represents a non-autoregressive UnitY T2U model."""

    model_dim: int
    encoder: Optional[TransformerEncoder]
    decoder_frontend: NARDecoderFrontend
    decoder: FeedForwardTransformer
    final_proj: Projection
    target_vocab_info: VocabularyInfo
    prosody_proj: Optional[Projection]

    def __init__(
        self,
        encoder: Optional[TransformerEncoder],
        decoder_frontend: NARDecoderFrontend,
        decoder: FeedForwardTransformer,
        final_proj: Projection,
        target_vocab_info: VocabularyInfo,
        prosody_proj: Optional[Projection] = None,
    ) -> None:
        super().__init__()

        self.model_dim = decoder.model_dim

        if encoder is not None:
            # if encoder.model_dim != self.model_dim:
            #     raise ValueError(
            #         f"`model_dim` of `encoder` and `model_dim` of `decoder` must be equal, but are {encoder.model_dim} and {self.model_dim} instead."
            #     )

            self.encoder = encoder
        # else:
        #     self.register_module("encoder", None)

        if decoder_frontend.model_dim != self.model_dim:
            raise ValueError(
                f"`model_dim` of `decoder_frontend` and `model_dim` of `decoder` must be equal, but are {decoder_frontend.model_dim} and {self.model_dim} instead."
            )

        self.decoder_frontend = decoder_frontend
        self.decoder = decoder

        self.final_proj = final_proj

        self.target_vocab_info = target_vocab_info

        self.prosody_proj = prosody_proj

    def forward(
        self,
        text_decoder_output: Tensor,
        text_decoder_padding_mask: Optional[PaddingMask],
        text_seqs: Optional[Tensor],
        duration_factor: float = 1.0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[SequenceModelOutput, Optional[PaddingMask], Tensor]:
        encoder_output, encoder_padding_mask = self.encode(
            text_decoder_output, text_decoder_padding_mask
        )

        if self.prosody_proj is not None and film_cond_emb is not None:
            encoder_output = encoder_output + self.prosody_proj(film_cond_emb)

        decoder_output, decoder_padding_mask, durations = self.decode(
            encoder_output,
            encoder_padding_mask,
            text_seqs,
            duration_factor,
            film_cond_emb,
        )

        return self.project(decoder_output), decoder_padding_mask, durations

    def encode(
        self,
        text_decoder_output: Tensor,
        text_decoder_padding_mask: Optional[PaddingMask],
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        if self.encoder is None:
            return text_decoder_output, text_decoder_padding_mask

        return self.encoder(text_decoder_output, text_decoder_padding_mask)  # type: ignore[no-any-return]

    def decode(
        self,
        encoder_output: Tensor,
        encoder_padding_mask: Optional[PaddingMask],
        text_seqs: Optional[Tensor],
        duration_factor: float = 1.0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor]:
        # encoder_output: (N, S, M)
        # text_seqs: (N, S)
        seqs, padding_mask, durations = self.decoder_frontend(
            encoder_output,
            encoder_padding_mask,
            text_seqs,
            duration_factor,
            film_cond_emb,
        )

        seqs, padding_mask = self.decoder(
            seqs, padding_mask, film_cond_emb=film_cond_emb
        )

        return seqs, padding_mask, durations  # type: ignore[no-any-return]

    def project(self, decoder_output: Tensor) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_info)


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
