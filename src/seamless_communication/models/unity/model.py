# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union, final, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import mse_loss, pad, log_softmax, ctc_loss
from fairseq2.data import VocabularyInfo
from fairseq2.models.encoder_decoder import EncoderDecoderModel
from fairseq2.models.sequence import SequenceModelOutput
from fairseq2.models.transformer.frontend import TransformerFrontend
from fairseq2.nn.incremental_state import IncrementalStateBag
from fairseq2.nn.padding import PaddingMask, apply_padding_mask
from fairseq2.nn.projection import Projection
from fairseq2.nn.transformer import TransformerDecoder, TransformerEncoder
from fairseq2.models.seq2seq import Seq2SeqBatch

from seamless_communication.models.generator.ecapa_tdnn import ECAPA_TDNN
from seamless_communication.models.unity.fft_decoder import FeedForwardTransformer
from seamless_communication.models.unity.nar_decoder_frontend import NARDecoderFrontend
from seamless_communication.train.utils.multi_task_configs import AuxMTLModel


@dataclass
class UnitYBatch(Seq2SeqBatch):
    """
    Holds a typical UnitY batch, extending the Seq2SeqBatch,
    the target_seqs/target_padding_mask refer to unit sequence.
    """

    prosody_input_seqs: Optional[Tensor] = None
    """The prosody input sequences. *Shape:* :math:`(N,S_{src},*)`, where :math:`N` is
    the batch size, :math:`S_{src}` is the sequence length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    prosody_input_padding_mask: Optional[PaddingMask] = None
    """The padding mask of ``prosody_input_seqs``. *Shape:* :math:`(N,S_{src})`, where
    :math:`N` is the batch size and :math:`S_{src}` is the sequence
    length."""

    target_text_seqs: Optional[Tensor] = None
    """The target text sequences. *Shape:* :math:`(N,S_{tgt},*)`, where :math:`N` is
    the batch size, :math:`S_{tgt}` is the target text length, and :math:`*`
    is any number of sequence-specific dimensions including none."""

    target_text_padding_mask: Optional[PaddingMask] = None
    """The padding mask of ``target_text_seqs``. *Shape:* :math:`(N,S_{tgt})`, where
    :math:`N` is the batch size and :math:`S_{tgt}` is the target text
    length."""

    duration_factor: Optional[float] = 1.0
    """The duration factor for UnitYNART2UModel"""

    def num_target_text_elements(self) -> int:
        """Return the number of elements in the target text sequences."""
        if self.target_text_padding_mask is None:
            return self.target_text_seqs.numel()

        return int(self.target_text_padding_mask.seq_lens.sum())

    def as_input_and_target(self) -> Tuple[Seq2SeqBatch, Tensor, Tensor]:
        """Use this batch for model training or validation.

        :returns:
          - A new batch with the target text/unit sequences trimmed one step
            from the end to use as model input.
          - The target text/unit sequences trimmed one step from the beginning
            to use in loss computation.
        """
        if (seq_len := self.target_seqs.size(1)) < 2:
            raise ValueError(
                f"The sequence length of `target_seqs` must be at least 2 for training, but is {seq_len} instead."
            )

        target_text_seqs = self.target_text_seqs[:, :-1]

        if self.target_text_padding_mask is None:
            target_text_padding_mask = None
        else:
            target_text_padding_mask = self.target_text_padding_mask.trim(1)

        batch = UnitYBatch(
            source_seqs=self.source_seqs,
            source_padding_mask=self.source_padding_mask,
            target_seqs=self.target_seqs,
            target_padding_mask=self.target_padding_mask,
            prosody_input_seqs=self.prosody_input_seqs,
            prosody_input_padding_mask=self.prosody_input_padding_mask,
            target_text_seqs=target_text_seqs,
            target_text_padding_mask=target_text_padding_mask,
            duration_factor=self.duration_factor,
        )

        return batch, self.target_text_seqs[:, 1:], self.target_seqs


@dataclass
class UnitYModelOutput:
    """Holds the output of the UnitY model."""

    text_output: SequenceModelOutput
    """Holds the text decoder output"""

    unit_output: SequenceModelOutput
    """Holds the unit decoder output"""

    log_durations: Tensor
    """Holds the duration predictor output (for duration loss)"""

    aux_mtl_output: Tensor
    """Holds the multi-task model output (for auxiliary loss)"""

    char_seqs: Tensor
    """Holds the character sequence (for auxiliary ctc loss)"""

    char_padding_mask: PaddingMask
    """Holds the padding mask for character sequence"""

    attn_hard_dur: Tensor
    """Holds the aligned hard duration b/t character and units"""

    attn_lprob: Tensor
    """Holds the aligned soft duration b/t character and units"""

    def compute_loss(
        self,
        text_targets: Tensor,
        unit_targets: Tensor,
        *,
        text_loss_weight: float = 1.0,
        aux_loss_type: Optional[str] = None,
        aux_loss_weight: float = 0.0,
        duration_loss_weight: float = 1.0,
        forward_sum_loss_weight: float = 1.0,
        ignore_text_prefix_size: int = 0,
        ignore_unit_prefix_size: int = 0,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        """
        Compute the UnitY multi-task training loss
        """
        unit_lens = (unit_targets != self.unit_output.vocab_info.pad_idx).sum(-1)
        char_lens = self.char_padding_mask.seq_lens

        loss = self.unit_output.compute_loss(
            unit_targets.long(),
            ignore_prefix_size=ignore_unit_prefix_size,
            label_smoothing=label_smoothing,
        )

        if text_loss_weight > 0.:
            text_nll_loss = self.text_output.compute_loss(
                text_targets,
                ignore_prefix_size=ignore_text_prefix_size,
                label_smoothing=label_smoothing,
            )
            loss += text_loss_weight * text_nll_loss

        # calculate duration loss
        if duration_loss_weight > 0.:
            log_durations = apply_padding_mask(self.log_durations, self.char_padding_mask)
            duration_target = torch.log(self.attn_hard_dur + 1)
            duration_loss = mse_loss(log_durations, duration_target, reduction="sum")
            loss += duration_loss_weight * duration_loss

        # calculate forward sum loss
        if forward_sum_loss_weight > 0.:
            # a row must be added to the attention matrix to account for blank token of CTC loss
            # (bsz, T_feat, T_text + 1)
            log_p_attn_pd = pad(self.attn_lprob, (1, 0, 0, 0, 0, 0), value=np.log(np.e**-1))

            forward_sum_loss = 0.
            for i in range(self.attn_lprob.size(0)):
                # every target is mapped to a unique position
                target_seq = torch.arange(1, char_lens[i] + 1).unsqueeze(0)

                # (T_feat, 1, T_text + 1)
                cur_log_p_attn_pd = log_p_attn_pd[i, :unit_lens[i], :char_lens[i] + 1].unsqueeze(1)
                cur_log_p_attn_pd = log_softmax(cur_log_p_attn_pd, dim=-1)

                forward_sum_loss += ctc_loss(
                    log_probs=cur_log_p_attn_pd.float(),  # for fp16
                    targets=target_seq,
                    input_lengths=unit_lens[i: i + 1],
                    target_lengths=char_lens[i: i + 1],
                    reduction="sum",
                    zero_infinity=True,
                )

            loss += forward_sum_loss_weight * forward_sum_loss

        # calculate auxiliary loss
        if aux_loss_weight > 0.:
            aux_lprobs = self.aux_mtl_output.log_softmax(dim=-1)
            if aux_loss_type == "ctc":
                with torch.backends.cudnn.flags(enabled=False):
                    aux_loss = ctc_loss(
                        aux_lprobs.transpose(0, 1),
                        self.char_seqs,
                        unit_lens,
                        char_lens,
                        reduction="sum",
                        zero_infinity=True,
                    )
            else:
                raise NotImplementedError

            loss += aux_loss_weight * aux_loss

        return loss


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
    aux_mtl_model: Optional[AuxMTLModel]

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
        target_vocab_info: VocabularyInfo,
        prosody_encoder_model: Optional[ECAPA_TDNN] = None,
        aux_mtl_model: Optional[AuxMTLModel] = None,
        input_modality: str = "speech",
    ) -> None:
        model_dim = speech_encoder.model_dim

        super().__init__(model_dim, target_vocab_info)

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

            self.register_module("text_decoder_frontend", None)
            self.register_module("text_decoder", None)
            self.register_module("final_proj", None)

        if t2u_model is not None:
            self.t2u_model = t2u_model
        else:
            self.register_module("t2u_model", None)

        self.target_vocab_info = target_vocab_info
        if prosody_encoder_model is not None:
            self.prosody_encoder_model = prosody_encoder_model
        else:
            self.register_module("prosody_encoder_model", None)

        if aux_mtl_model is not None:
            self.aux_mtl_model = aux_mtl_model
        else:
            self.register_module("aux_mtl_model", None)

    @final
    def inference_trim(self):
        """Trim model parameters for inference-only"""
        self.aux_mtl_model = None
        if isinstance(self.t2u_model, UnitYNART2UModel):
            self.t2u_model.decoder_frontend.alignment_encoder = None

    def set_num_updates(self, num_updates: int):
        """Set the number of parameters updates."""
        if self.t2u_model.decoder_frontend.alignment_encoder is not None:
            self.t2u_model.decoder_frontend.alignment_encoder.set_num_updates(num_updates)

    @final
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

    @final
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

    @final
    def project(
        self, decoder_output: Tensor, decoder_padding_mask: Optional[PaddingMask]
    ) -> SequenceModelOutput:
        if self.final_proj is None:
            raise ValueError(
                "`project()` requires a final_proj layer, but the current UnitY model does not have one."
            )

        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_info)

    @final
    def forward(self, batch: UnitYBatch) -> UnitYModelOutput:
        """
        Used only during training, inference forward is at UnitYGenerator
        """
        speech_enc_out, speech_enc_padding_mask = self.encode(
            batch.source_seqs, batch.source_padding_mask
        )

        text_dec_out, text_dec_padding_mask = self.decode(
            batch.target_text_seqs,
            batch.target_text_padding_mask,
            speech_enc_out,
            speech_enc_padding_mask,
        )

        text_output = self.project(text_dec_out, text_dec_padding_mask)

        # forward prosody_encoder_model
        prosody_encoder_out = None
        if self.prosody_encoder_model is not None:
            prosody_input_seqs = batch.prosody_input_seqs
            if prosody_input_seqs is None:
                prosody_input_seqs = batch.source_seqs

            prosody_encoder_out = self.prosody_encoder_model(
                prosody_input_seqs,
                batch.source_padding_mask,  # always the same
            ).unsqueeze(1)

        # forward t2u model
        assert isinstance(self.t2u_model, UnitYNART2UModel), "only NAR T2U is supported in training"
        unit_output, unit_dec_padding_mask, log_durations, inner_states, char_seqs, char_padding_mask, attn_lprob, attn_hard_dur = self.t2u_model(
            text_dec_out,
            text_dec_padding_mask,
            batch.target_text_seqs,
            batch.target_seqs,
            batch.duration_factor,
            prosody_encoder_out,
        )

        if self.aux_mtl_model is None:
            aux_mtl_output = None
        else:
            aux_mtl_output = self.aux_mtl_model(inner_states)

        return UnitYModelOutput(
            text_output=text_output,
            unit_output=unit_output,
            log_durations=log_durations,
            aux_mtl_output=aux_mtl_output,
            char_seqs=char_seqs,
            char_padding_mask=char_padding_mask,
            attn_lprob=attn_lprob,
            attn_hard_dur=attn_hard_dur,
        )


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

    @final
    def encode(
        self, seqs: Tensor, padding_mask: Optional[PaddingMask]
    ) -> Tuple[Tensor, Optional[PaddingMask]]:
        seqs, padding_mask = self.encoder_frontend(seqs, padding_mask)
        return self.encoder(seqs, padding_mask)  # type: ignore[no-any-return]

    @final
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

    @final
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

        self.target_vocab_info = target_vocab_info

        self.prosody_proj = prosody_proj

    def forward(
        self,
        text_decoder_output: Tensor,
        text_decoder_padding_mask: Optional[PaddingMask],
        text_seqs: Optional[Tensor],
        unit_seqs: Optional[Tensor] = None,
        duration_factor: float = 1.0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[SequenceModelOutput, Optional[PaddingMask], Tensor, List[Tensor], Tensor, PaddingMask, Tensor, Tensor]:
        encoder_output, encoder_padding_mask = self.encode(
            text_decoder_output, text_decoder_padding_mask
        )

        if self.prosody_proj is not None and film_cond_emb is not None:
            encoder_output = encoder_output + self.prosody_proj(film_cond_emb)

        decoder_output, decoder_padding_mask, log_durations, inner_states, char_seqs, char_padding_mask, attn_lprob, attn_hard_dur = self.decode(
            encoder_output,
            encoder_padding_mask,
            text_seqs,
            unit_seqs=unit_seqs,
            duration_factor=duration_factor,
            film_cond_emb=film_cond_emb,
        )

        return self.project(decoder_output), decoder_padding_mask, log_durations, inner_states, char_seqs, char_padding_mask, attn_lprob, attn_hard_dur

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
        unit_seqs: Optional[Tensor],
        duration_factor: float = 1.0,
        film_cond_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[PaddingMask], Tensor, List[Tensor], Tensor, PaddingMask, Tensor, Tensor]:
        # encoder_output: (N, S, M)
        # text_seqs: (N, S)
        seqs, padding_mask, log_durations, char_seqs, char_padding_mask, attn_lprob, attn_hard_dur = self.decoder_frontend(
            encoder_output,
            encoder_padding_mask,
            text_seqs,
            unit_seqs,
            duration_factor,
            film_cond_emb,
        )

        seqs, padding_mask, inner_states = self.decoder(
            seqs, padding_mask, film_cond_emb=film_cond_emb
        )

        return seqs, padding_mask, log_durations, inner_states, char_seqs, char_padding_mask, attn_lprob, attn_hard_dur # type: ignore[no-any-return]

    def project(self, decoder_output: Tensor) -> SequenceModelOutput:
        logits = self.final_proj(decoder_output)

        return SequenceModelOutput(logits, self.target_vocab_info)
