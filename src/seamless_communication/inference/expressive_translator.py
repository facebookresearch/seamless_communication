# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import torch
import torchaudio
from fairseq2.assets.card import AssetCard
from fairseq2.data import SequenceData, StringLike
from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.nn.padding import apply_padding_mask, get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device
from torch.nn import Module

from seamless_communication.inference.generator import SequenceGeneratorOptions
from seamless_communication.inference.pretssel_generator import PretsselGenerator
from seamless_communication.inference.translator import BatchedSpeechOutput, Translator
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)

AUDIO_SAMPLE_RATE = 16000


class ExpressiveTranslator(Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        vocoder_name_or_card: Union[str, AssetCard, None],
        device: Device,
        dtype: DataType,
    ):
        super().__init__()

        unit_tokenizer = load_unity_unit_tokenizer(model_name_or_card)

        self.translator = Translator(
            model_name_or_card,
            vocoder_name_or_card=None,
            device=device,
            dtype=dtype,
        )

        self.pretssel_generator = PretsselGenerator(
            vocoder_name_or_card,
            vocab_info=unit_tokenizer.vocab_info,
            device=device,
            dtype=dtype,
        )

        self.fbank_extractor = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=False,
            device=device,
            dtype=dtype,
        )

        _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(vocoder_name_or_card)
        self.gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
        self.gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

    @staticmethod
    def remove_prosody_tokens_from_text(text_output: List[str]) -> List[str]:
        modified_text_output = []
        for text in text_output:
            # filter out prosody tokens, there is only emphasis '*', and pause '='
            text = str(text).replace("*", "").replace("=", "")
            text = " ".join(text.split())
            modified_text_output.append(text)
        return modified_text_output

    @torch.inference_mode()
    def predict(
        self,
        input: Union[Path, SequenceData],
        tgt_lang: str,
        text_generation_opts: Optional[SequenceGeneratorOptions] = None,
        unit_generation_opts: Optional[SequenceGeneratorOptions] = None,
        unit_generation_ngram_filtering: bool = False,
        duration_factor: float = 1.0,
    ) -> Tuple[List[StringLike], Optional[BatchedSpeechOutput]]:
        """
        The main method used to perform inference on all tasks.

        :param input:
            Either path to audio or audio Tensor.
        :param tgt_lang:
            Target language to decode into.
        :param text_generation_opts:
            Text generation hyperparameters for incremental decoding.
        :param unit_generation_opts:
            Unit generation hyperparameters for incremental decoding.
        :param unit_generation_ngram_filtering:
            If True, removes consecutive repeated ngrams
            from the decoded unit output.

        :returns:
            - Batched list of Translated text.
            - Translated BatchedSpeechOutput.
        """
        if isinstance(input, dict):
            src = cast(SequenceData, input)
            src_gcmvn = deepcopy(src)

            fbank, padding_mask = get_seqs_and_padding_mask(src)
            gcmvn_fbank = fbank.subtract(self.gcmvn_mean).divide(self.gcmvn_std)
            # due to padding, batched std_mean calculation is wrong
            mean = torch.zeros_like(fbank[:, 0])
            std = torch.zeros_like(fbank[:, 0])
            for i, (i_fbank, i_seq_len) in enumerate(zip(fbank, src["seq_lens"])):
                std[i], mean[i] = torch.std_mean(i_fbank[:i_seq_len], dim=0)

            ucmvn_fbank = fbank.subtract(mean.unsqueeze(1)).divide(std.unsqueeze(1))
            src["seqs"] = apply_padding_mask(ucmvn_fbank, padding_mask)
            src_gcmvn["seqs"] = apply_padding_mask(gcmvn_fbank, padding_mask)

        elif isinstance(input, Path):
            # TODO: Replace with fairseq2.data once re-sampling is implemented.
            wav, sample_rate = torchaudio.load(input)
            wav = torchaudio.functional.resample(
                wav,
                orig_freq=sample_rate,
                new_freq=AUDIO_SAMPLE_RATE,
            )
            wav = wav.transpose(0, 1)

            data = self.fbank_extractor(
                {
                    "waveform": wav,
                    "sample_rate": AUDIO_SAMPLE_RATE,
                }
            )

            fbank = data["fbank"]
            gcmvn_fbank = fbank.subtract(self.gcmvn_mean).divide(self.gcmvn_std)
            std, mean = torch.std_mean(fbank, dim=0)
            fbank = fbank.subtract(mean).divide(std)

            src = SequenceData(
                seqs=fbank.unsqueeze(0),
                seq_lens=torch.LongTensor([fbank.shape[0]]),
                is_ragged=False,
            )
            src_gcmvn = SequenceData(
                seqs=gcmvn_fbank.unsqueeze(0),
                seq_lens=torch.LongTensor([gcmvn_fbank.shape[0]]),
                is_ragged=False,
            )

        text_output, unit_output = self.translator.predict(
            src,
            "s2st",
            tgt_lang,
            text_generation_opts=text_generation_opts,
            unit_generation_opts=unit_generation_opts,
            unit_generation_ngram_filtering=unit_generation_ngram_filtering,
            duration_factor=duration_factor,
            prosody_encoder_input=src_gcmvn,
        )
        text_output = self.remove_prosody_tokens_from_text(text_output)

        assert unit_output is not None
        speech_output = self.pretssel_generator.predict(
            unit_output.units,
            tgt_lang=tgt_lang,
            prosody_encoder_input=src_gcmvn,
        )
        return text_output, speech_output
