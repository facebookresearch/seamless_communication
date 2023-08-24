# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text.text_tokenizer import TextTokenizer
from fairseq2.data.typing import StringLike
from fairseq2.generation import SequenceToTextOutput, SequenceGeneratorOptions
from fairseq2.memory import MemoryBlock
from fairseq2.typing import DataType, Device
from torch import Tensor
from enum import Enum, auto
from seamless_communication.models.inference.ngram_repeat_block_processor import (
    NGramRepeatBlockProcessor,
)

from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYGenerator,
    UnitYModel,
    load_unity_model,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)
from seamless_communication.models.unity.generator import SequenceToUnitOutput
from seamless_communication.models.vocoder import load_vocoder_model, Vocoder


class Task(Enum):
    S2ST = auto()
    S2TT = auto()
    T2ST = auto()
    T2TT = auto()
    ASR = auto()


class Modality(Enum):
    SPEECH = "speech"
    TEXT = "text"


class Translator(nn.Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        vocoder_name_or_card: Union[str, AssetCard],
        device: Device,
        dtype: DataType,
    ):
        super().__init__()
        # Load the model.
        self.model: UnitYModel = self.load_model_for_inference(
            load_unity_model, model_name_or_card, device, dtype
        )
        self.text_tokenizer = load_unity_text_tokenizer(model_name_or_card)
        self.unit_tokenizer = load_unity_unit_tokenizer(model_name_or_card)
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
            pad_idx=self.text_tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )
        # Load the vocoder.
        self.vocoder: Vocoder = self.load_model_for_inference(
            load_vocoder_model, vocoder_name_or_card, device, torch.float32
        )

    @staticmethod
    def load_model_for_inference(
        load_model_fn: Callable[..., nn.Module],
        model_name_or_card: Union[str, AssetCard],
        device: Device,
        dtype: DataType,
    ) -> nn.Module:
        model = load_model_fn(model_name_or_card, device=device, dtype=dtype)
        model.eval()
        return model

    @classmethod
    def get_prediction(
        cls,
        model: UnitYModel,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: UnitTokenizer,
        src: Dict[str, Tensor],
        input_modality: Modality,
        output_modality: Modality,
        tgt_lang: str,
        ngram_filtering: bool = False,
    ) -> Tuple[SequenceToTextOutput, Optional[SequenceToUnitOutput]]:
        if input_modality == Modality.TEXT:
            # need to adjust this since src_len is smaller for text.
            max_len_a = 25
        else:
            max_len_a = 1
        text_opts = SequenceGeneratorOptions(beam_size=5, soft_max_seq_len=(1, 200))
        unit_opts = SequenceGeneratorOptions(
            beam_size=5, soft_max_seq_len=(max_len_a, 50)
        )
        if ngram_filtering:
            text_opts.logits_processor = NGramRepeatBlockProcessor(
                no_repeat_ngram_size=4
            )
            unit_opts.logits_processor = NGramRepeatBlockProcessor(
                no_repeat_ngram_size=4
            )
        generator = UnitYGenerator(
            model,
            text_tokenizer,
            tgt_lang,
            unit_tokenizer if output_modality == Modality.SPEECH else None,
            text_opts=text_opts,
            unit_opts=unit_opts,
        )
        return generator(
            src["seqs"],
            src["seq_lens"],
            input_modality.value,
            output_modality.value,
            ngram_filtering=ngram_filtering,
        )

    def get_modalities_from_task(self, task: Task) -> Tuple[Modality, Modality]:
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
        input: Union[str, torch.Tensor],
        task_str: str,
        tgt_lang: str,
        src_lang: Optional[str] = None,
        spkr: Optional[int] = -1,
        ngram_filtering: bool = False,
        sample_rate: int = 16000,
    ) -> Tuple[StringLike, Optional[List[Tensor]], Optional[int]]:
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
        :param spkr:
            Speaker id for vocoder.

        :returns:
            - Translated text.
            - Generated output audio waveform corresponding to the translated text.
            - Sample rate of output audio waveform.
        """
        try:
            task = Task[task_str.upper()]
        except KeyError:
            raise ValueError(f"Unsupported task: {task_str}")

        input_modality, output_modality = self.get_modalities_from_task(task)

        if input_modality == Modality.SPEECH:
            audio = input
            if isinstance(audio, str):
                with Path(audio).open("rb") as fb:
                    block = MemoryBlock(fb.read())
                decoded_audio = self.decode_audio(block)
            else:
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
            self.token_encoder = self.text_tokenizer.create_encoder(
                task="translation", lang=src_lang, mode="source", device=self.device
            )
            src = self.collate(self.token_encoder(text))

        result = self.get_prediction(
            self.model,
            self.text_tokenizer,
            self.unit_tokenizer,
            src,
            input_modality,
            output_modality,
            tgt_lang=tgt_lang,
            ngram_filtering=ngram_filtering,
        )

        text_out = result[0]
        unit_out = result[1]
        if output_modality == Modality.TEXT:
            return text_out.sentences[0], None, None
        else:
            units = unit_out.units[:, 1:][0].cpu().numpy().tolist()
            wav_out = self.vocoder(units, tgt_lang, spkr, dur_prediction=True)
            return text_out.sentences[0], wav_out, sample_rate
