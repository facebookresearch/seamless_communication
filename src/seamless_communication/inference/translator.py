# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from fairseq2.assets import asset_store
from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater, SequenceData, StringLike
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import TextTokenizer
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import PaddingMask, get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device
from torch import Tensor

from seamless_communication.inference.generator import (
    SequenceGeneratorOptions,
    UnitYGenerator,
)
from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYModel,
    UnitYNART2UModel,
    UnitYT2UModel,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
    unity_archs,
)
from seamless_communication.models.vocoder import load_vocoder_model
from seamless_communication.toxicity import (
    ETOXBadWordChecker,
    load_etox_bad_word_checker,
)
from seamless_communication.toxicity.mintox import mintox_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class Task(Enum):
    S2ST = auto()
    S2TT = auto()
    T2ST = auto()
    T2TT = auto()
    ASR = auto()


class Modality(Enum):
    SPEECH = "speech"
    TEXT = "text"


@dataclass
class BatchedSpeechOutput:
    units: List[List[int]]
    """The batched list of generated units."""

    audio_wavs: List[Tensor]
    """The batched list of audio waveforms."""

    sample_rate: int = 16000
    """Sample rate of the audio waveforms."""


class Translator(nn.Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        vocoder_name_or_card: Union[str, AssetCard, None],
        device: Device,
        text_tokenizer: Optional[TextTokenizer] = None,
        apply_mintox: bool = False,
        dtype: DataType = torch.float16,
        input_modality: Optional[Modality] = None,
        output_modality: Optional[Modality] = None,
    ):
        super().__init__()

        if isinstance(model_name_or_card, str):
            model_name_or_card = asset_store.retrieve_card(model_name_or_card)

        assert isinstance(model_name_or_card, AssetCard)

        if input_modality or output_modality:
            unity_config = unity_archs.get_config(
                model_name_or_card.field("model_arch").as_(str)
            )
            # Skip loading the text encoder.
            if input_modality == Modality.SPEECH:
                unity_config.use_text_encoder = False
            # Skip loading the T2U model.
            if output_modality == Modality.TEXT:
                unity_config.t2u_config = None
            model_name_or_card.field("model_config").set(unity_config)

        # Load the model.
        if device == torch.device("cpu"):
            dtype = torch.float32

        self.model = load_unity_model(model_name_or_card, device=device, dtype=dtype)
        self.model.eval()
        assert isinstance(self.model, UnitYModel)

        if text_tokenizer is None:
            self.text_tokenizer: TextTokenizer = load_unity_text_tokenizer(
                model_name_or_card
            )
        else:
            self.text_tokenizer = text_tokenizer

        self.unit_tokenizer: Optional[UnitTokenizer] = None
        if self.model.t2u_model is not None:
            self.unit_tokenizer = load_unity_unit_tokenizer(model_name_or_card)

        self.bad_word_checker: Optional[ETOXBadWordChecker] = None
        if apply_mintox:
            self.bad_word_checker = load_etox_bad_word_checker("mintox")

        self.apply_mintox = apply_mintox

        self.device = device
        self.decode_audio = AudioDecoder(dtype=torch.float32, device=device)
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
            device=device,
            dtype=dtype,
        )
        self.collate = Collater(
            pad_value=self.text_tokenizer.vocab_info.pad_idx or 0, pad_to_multiple=2
        )
        self.vocoder = None
        if vocoder_name_or_card is not None and (
            output_modality is None or output_modality == Modality.SPEECH
        ):
            self.vocoder = load_vocoder_model(
                vocoder_name_or_card, device=device, dtype=dtype
            )
            self.vocoder.eval()

    @classmethod
    def get_prediction(
        cls,
        model: UnitYModel,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: Optional[UnitTokenizer],
        seqs: Tensor,
        padding_mask: Optional[PaddingMask],
        input_modality: Modality,
        output_modality: Modality,
        tgt_lang: str,
        text_generation_opts: SequenceGeneratorOptions,
        unit_generation_opts: Optional[SequenceGeneratorOptions],
        unit_generation_ngram_filtering: bool = False,
        duration_factor: float = 1.0,
        prosody_encoder_input: Optional[SequenceData] = None,
    ) -> Tuple[List[StringLike], Optional[Tensor]]:
        # We disregard unit generations opts for the NAR T2U decoder.
        if output_modality != Modality.SPEECH or isinstance(
            model.t2u_model, UnitYNART2UModel
        ):
            unit_generation_opts = None

        generator = UnitYGenerator(
            model,
            text_tokenizer,
            tgt_lang,
            unit_tokenizer if output_modality == Modality.SPEECH else None,
            text_opts=text_generation_opts,
            unit_opts=unit_generation_opts,
        )

        return generator(
            seqs,
            padding_mask,
            input_modality.value,
            output_modality.value,
            ngram_filtering=unit_generation_ngram_filtering,
            duration_factor=duration_factor,
            prosody_encoder_input=prosody_encoder_input,
        )

    @staticmethod
    def get_modalities_from_task_str(task_str: str) -> Tuple[Modality, Modality]:
        try:
            task = Task[task_str.upper()]
        except KeyError:
            raise ValueError(f"Unsupported task: {task_str}")

        if task == Task.S2ST:
            return Modality.SPEECH, Modality.SPEECH
        # ASR is treated as S2TT with src_lang == tgt_lang
        elif task == Task.S2TT or task == Task.ASR:
            return Modality.SPEECH, Modality.TEXT
        elif task == Task.T2TT:
            return Modality.TEXT, Modality.TEXT
        else:
            return Modality.TEXT, Modality.SPEECH

    @torch.inference_mode()
    def predict(
        self,
        input: Union[str, Tensor, SequenceData],
        task_str: str,
        tgt_lang: str,
        src_lang: Optional[str] = None,
        text_generation_opts: Optional[SequenceGeneratorOptions] = None,
        unit_generation_opts: Optional[SequenceGeneratorOptions] = None,
        spkr: Optional[int] = -1,
        sample_rate: int = 16000,
        unit_generation_ngram_filtering: bool = False,
        duration_factor: float = 1.0,
        prosody_encoder_input: Optional[SequenceData] = None,
        src_text: Optional[StringLike] = None,
    ) -> Tuple[List[StringLike], Optional[BatchedSpeechOutput]]:
        """
        The main method used to perform inference on all tasks.

        :param input:
            Either text or path to audio or audio Tensor.
        :param task_str:
            String representing the task.
            Valid choices are "S2ST", "S2TT", "T2ST", "T2TT", "ASR"
        :param tgt_lang:
            Target language to decode into.
        :param src_lang:
            Source language of input, only required for T2ST, T2TT tasks.
        :param text_generation_opts:
            Text generation hyperparameters for incremental decoding.
        :param unit_generation_opts:
            Unit generation hyperparameters for incremental decoding.
        :param spkr:
            Speaker id for vocoder.
        :param unit_generation_ngram_filtering:
            If True, removes consecutive repeated ngrams
            from the decoded unit output.
        :param src_text:
            Optional source transcript (obtained by ASR for instance). This is used for
            applying mintox toxicity mitigation. If this is not specify and apply_mintox=True
            then src_lang must be specified and ASR will be run on the audio source.

        :returns:
            - Batched list of Translated text.
            - Translated BatchedSpeechOutput.
        """
        input_modality, output_modality = self.get_modalities_from_task_str(task_str)

        if self.apply_mintox and not (src_lang is not None or src_text is not None):
            raise ValueError(
                "`src_lang` must be specified when `apply_mintox` is `True` or you need to specify src_text."
            )

        if isinstance(input, dict):
            src = cast(SequenceData, input)
        elif input_modality == Modality.SPEECH:
            audio = input
            if isinstance(audio, str):
                with Path(audio).open("rb") as fb:
                    block = MemoryBlock(fb.read())
                decoded_audio = self.decode_audio(block)
            else:
                assert (
                    audio.dim() <= 2
                ), "The audio tensor can't be more than 2 dimensions."
                if audio.dim() == 1:
                    audio = audio.unsqueeze(1)
                elif audio.dim() == 2 and audio.size(0) < audio.size(1):
                    logger.warning(
                        "Transposing audio tensor from (bsz, seq_len) -> (seq_len, bsz)."
                    )
                    audio = audio.transpose(0, 1)

                decoded_audio = {
                    "waveform": audio,
                    "sample_rate": sample_rate,
                    "format": -1,
                }
            src = self.collate(self.convert_to_fbank(decoded_audio))["fbank"]
        else:
            if src_lang is None:
                raise ValueError("src_lang must be specified for T2ST, T2TT tasks.")

            text = input
            assert isinstance(text, str)

            self.token_encoder = self.text_tokenizer.create_encoder(
                task="translation", lang=src_lang, mode="source", device=self.device
            )
            src = self.collate(self.token_encoder(text))

        assert isinstance(self.model, UnitYModel)

        seqs, padding_mask = get_seqs_and_padding_mask(src)

        if text_generation_opts is None:
            text_generation_opts = SequenceGeneratorOptions(
                beam_size=5, soft_max_seq_len=(1, 200)
            )
        if unit_generation_opts is None:
            unit_generation_opts = SequenceGeneratorOptions(
                beam_size=5, soft_max_seq_len=(25, 50)
            )

        texts, units = self.get_prediction(
            self.model,
            self.text_tokenizer,
            self.unit_tokenizer,
            seqs,
            padding_mask,
            input_modality,
            output_modality,
            tgt_lang,
            text_generation_opts,
            unit_generation_opts,
            unit_generation_ngram_filtering=unit_generation_ngram_filtering,
            duration_factor=duration_factor,
            prosody_encoder_input=prosody_encoder_input,
        )

        if self.apply_mintox and task_str != Task.ASR.name:
            if input_modality == Modality.SPEECH:
                if src_text is not None:
                    src_texts = [src_text]
                else:
                    src_texts, _, = self.predict(
                        input=input,
                        task_str=Task.ASR.name,
                        tgt_lang=tgt_lang,
                        src_lang=src_lang,
                        text_generation_opts=text_generation_opts,
                        unit_generation_opts=unit_generation_opts,
                        spkr=spkr,
                        sample_rate=sample_rate,
                        unit_generation_ngram_filtering=unit_generation_ngram_filtering,
                    )
            else:
                assert isinstance(input, str)

                src_texts = [input]

            assert src_lang is not None
            assert self.unit_tokenizer is not None
            assert self.bad_word_checker is not None

            texts, units = mintox_pipeline(
                model=self.model,
                text_tokenizer=self.text_tokenizer,
                unit_tokenizer=self.unit_tokenizer,
                device=self.device,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                model_input=src,
                input_modality=input_modality,
                output_modality=output_modality,
                src_texts=src_texts,
                original_texts=texts,
                original_units=units,
                unit_generation_ngram_filtering=unit_generation_ngram_filtering,
                text_generation_opts=text_generation_opts,
                unit_generation_opts=unit_generation_opts,
                bad_word_checker=self.bad_word_checker,
                duration_factor=duration_factor,
                prosody_encoder_input=prosody_encoder_input,
            )

        if output_modality == Modality.TEXT:
            return texts, None
        else:
            assert units is not None

            if isinstance(self.model.t2u_model, UnitYT2UModel):
                # Remove the lang token for AR UnitY since the vocoder doesn't need it
                # in the unit sequence. tgt_lang is fed as an argument to the vocoder.
                units = units[:, 1:]
                duration_prediction = True
            else:
                # Vocoder duration predictions not required since the NAR
                # T2U model already predicts duration in the units.
                duration_prediction = False

            audio_wavs = []
            speech_units = []
            for i in range(len(units)):
                assert self.model.t2u_model is not None
                unit_padding_mask = (
                    units[i] != self.model.t2u_model.target_vocab_info.pad_idx
                )
                u = units[i][unit_padding_mask]
                speech_units.append(u.tolist())

            if self.vocoder is not None:
                translated_audio_wav = self.vocoder(
                    units, tgt_lang, spkr, dur_prediction=duration_prediction
                )
                for i in range(len(units)):
                    padding_removed_audio_wav = translated_audio_wav[
                        i,
                        :,
                        : int(
                            translated_audio_wav.size(-1)
                            * len(speech_units[i])
                            / len(units[i])
                        ),
                    ].unsqueeze(0)
                    audio_wavs.append(padding_removed_audio_wav)
            return (
                texts,
                BatchedSpeechOutput(
                    units=speech_units,
                    audio_wavs=audio_wavs,
                    sample_rate=sample_rate,
                ),
            )
