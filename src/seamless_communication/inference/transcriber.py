# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

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
from seamless_communication.models.inference.ngram_repeat_block_processor import (
    NGramRepeatBlockProcessor,
)

from seamless_communication.models.unity import (
    UnitYGenerator,
    UnitYModel,
    load_unity_model,
    load_unity_text_tokenizer,
)
from seamless_communication.models.unity.generator import SequenceToUnitOutput


class Transcriber(nn.Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        device: Device,
        dtype: DataType = torch.float16,
    ):
        super().__init__()
        # Load the model.
        if device == torch.device("cpu"):
            dtype = torch.float32
        self.model: UnitYModel = self.load_model_for_inference(
            load_unity_model, model_name_or_card, device, dtype
        )
        self.text_tokenizer = load_unity_text_tokenizer(model_name_or_card)
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
        src: Dict[str, Tensor],
        src_lang: str,
        ngram_filtering: bool = False,
        text_max_len_a: int = 1,
        text_max_len_b: int = 200,
    ) -> Tuple[SequenceToTextOutput, Optional[SequenceToUnitOutput]]:
        text_opts = SequenceGeneratorOptions(
            beam_size=5, soft_max_seq_len=(text_max_len_a, text_max_len_b)
        )

        if ngram_filtering:
            text_opts.logits_processor = NGramRepeatBlockProcessor(
                no_repeat_ngram_size=4
            )
        generator = UnitYGenerator(
            model,
            text_tokenizer,
            src_lang,
            text_opts=text_opts,
        )
        return generator(
            src["seqs"],
            src["seq_lens"],
            input_modality="speech",
            output_modality="text",
            ngram_filtering=ngram_filtering,
        )

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, Tensor],
        src_lang: str,
        ngram_filtering: bool = False,
        sample_rate: int = 16000,
        text_max_len_a: int = 1,
        text_max_len_b: int = 200,
    ) -> StringLike:
        """
        The main method used to perform transcription.

        :param audio:
            Either path to audio or audio Tensor.
        :param src_lang:
            Source language of audio.

        :returns:
            - Transcribed text.
        """
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
        if src_lang is None:
            raise ValueError("src_lang must be specified for transcription tasks.")

        result = self.get_prediction(
            self.model,
            self.text_tokenizer,
            src,
            src_lang=src_lang,
            ngram_filtering=ngram_filtering,
            text_max_len_a=text_max_len_a,
            text_max_len_b=text_max_len_b,
        )

        text_out = result[0]
        return text_out.sentences[0]
