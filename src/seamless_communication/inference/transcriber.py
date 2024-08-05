# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from fairseq2.assets import asset_store, download_manager
from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater
from fairseq2.data.audio import (
    AudioDecoder,
    AudioDecoderOutput,
    WaveformToFbankConverter,
)
from fairseq2.generation import BeamSearchSeq2SeqGenerator, Seq2SeqGeneratorOutput
from fairseq2.memory import MemoryBlock
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.nn.transformer.multihead_attention import AttentionWeightHook
from fairseq2.typing import DataType, Device
from scipy.signal import medfilt2d
from torch import Tensor

from seamless_communication.denoise.demucs import Demucs, DenoisingConfig
from seamless_communication.models.tokenizer import SPMTokenizer
from seamless_communication.models.unity import (
    UnitYX2TModel,
    load_unity_model,
    load_unity_text_tokenizer,
)
from seamless_communication.segment.silero_vad import SileroVADSegmenter


class EncDecAttentionsCollect(AttentionWeightHook):
    def __init__(self):
        super().__init__()
        self.attn_scores = []

    def __call__(self, m, attn, attn_weights) -> None:
        if attn_weights.shape[-2] > 1:
            val = torch.clone(attn_weights).detach().sum(dim=0).squeeze(0).tolist()
            self.attn_scores.extend(val)
        else:
            val = (
                torch.clone(attn_weights)
                .detach()
                .sum(dim=0)
                .sum(dim=0)
                .squeeze(0)
                .tolist()
            )
            self.attn_scores.append(val)

    def reset(self):
        self.attn_scores = []


@dataclass
class TranscriptionTokenStats:
    text: str
    time_s: float
    scores: List[float]


@dataclass
class TranscriptionToken:
    text: str
    time_s: float
    prob: float


class Transcription:
    text: str
    tokens: List[TranscriptionToken]

    def __init__(self, tokens: List[TranscriptionToken]):
        self.text = " ".join([t.text for t in tokens])
        self.tokens = tokens

    def __add__(self, other: "Transcription") -> "Transcription":
        self.text += " " + other.text
        self.tokens += other.tokens
        return self

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Transcriber(nn.Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.tokenizer = self.load_tokenizer(model_name_or_card)

        model = self.load_model_for_inference(
            load_unity_model, model_name_or_card, device, dtype
        )

        self.s2t = UnitYX2TModel(
            encoder_frontend=model.speech_encoder_frontend,
            encoder=model.speech_encoder,
            decoder_frontend=model.text_decoder_frontend,
            decoder=model.text_decoder,
            final_proj=model.final_proj,
            target_vocab_info=self.tokenizer.vocab_info,
        )

        self.enc_dec_attn_collector = EncDecAttentionsCollect()
        self.s2t.decoder.layers[-1].encoder_decoder_attn.register_attn_weight_hook(
            self.enc_dec_attn_collector
        )

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
            pad_value=self.tokenizer.vocab_info.pad_idx, pad_to_multiple=2
        )

    @staticmethod
    def load_tokenizer(
        model_name_or_card: Union[AssetCard, str]
    ) -> Union[SPMTokenizer, NllbTokenizer]:
        if isinstance(model_name_or_card, AssetCard):
            model_card = model_name_or_card
        else:
            model_card = asset_store.retrieve_card(model_name_or_card)

        tokenizer_type = model_card.field("tokenizer_type").as_(str)

        if tokenizer_type == "nllb":
            return load_unity_text_tokenizer(model_name_or_card)

        if tokenizer_type == "plain_spm":
            tokenizer_uri = model_card.field("tokenizer").as_(str)
            tokenizer_langs = model_card.field("langs").as_(list)
            tokenizer_path = download_manager.download_tokenizer(
                tokenizer_uri, model_name=""
            )
            return SPMTokenizer(pathname=tokenizer_path, langs=tokenizer_langs)
        raise NotImplementedError(f"Unknow tokenizer type '{tokenizer_type}'")

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

    @staticmethod
    def generate_lis(arr: List[Tuple[int, int]]) -> Tuple[int, List[Tuple[int, int]]]:
        n = len(arr)
        lis = [1] * n
        prev = [0] * n
        for i in range(0, n):
            prev[i] = i
        for i in range(1, n):
            for j in range(0, i):
                if arr[i] > arr[j] and lis[i] < lis[j] + 1:
                    lis[i] = lis[j] + 1
                    prev[i] = j
        maximum = 0
        idx = 0
        for i in range(n):
            if maximum < lis[i]:
                maximum = lis[i]
                idx = i
        seq = [arr[idx]]
        while idx != prev[idx]:
            idx = prev[idx]
            seq.append(arr[idx])
        return (maximum, list(reversed(seq)))

    @classmethod
    def _extract_timestamps(
        cls,
        attn_weights,
        audio_len,
        filter_width,
    ) -> List[float]:
        attn_weights = [attn_line[1:-1] for attn_line in attn_weights][1:]

        num_out_tokens = len(attn_weights)
        num_encoder_steps = len(attn_weights[0])
        attn_weights = np.array(attn_weights)
        attn_weights = attn_weights / attn_weights.sum(axis=0, keepdims=1)  # normalize
        attn_weights = medfilt2d(attn_weights, kernel_size=(filter_width, filter_width))

        # find timestamps using longest increasing subsequence algo
        col_maxes = np.argmax(attn_weights, axis=0)
        lis_input = [
            (out_tok_idx, -enc_bin_idx)
            for enc_bin_idx, out_tok_idx in enumerate(col_maxes)
        ]
        tok_idx_to_start_enc_bin_idx = {
            out_tok_idx: -enc_bin_idx
            for out_tok_idx, enc_bin_idx in cls.generate_lis(lis_input)[1]
        }
        prev_start = 0
        starts = []
        for tok_idx in range(num_out_tokens):
            start_enc_bin_idx = tok_idx_to_start_enc_bin_idx.get(tok_idx, prev_start)
            starts.append(start_enc_bin_idx)
            prev_start = start_enc_bin_idx
        seconds_per_enc_pos = audio_len / num_encoder_steps
        start_times = [seconds_per_enc_pos * start_pos for start_pos in starts]
        return start_times

    @classmethod
    def _collect_word_level_stats(
        cls, pieces: List[str], token_timestamps: List[float], step_scores: List[float]
    ) -> List[TranscriptionToken]:
        assert len(pieces) == len(token_timestamps) and len(token_timestamps) == len(
            step_scores
        )
        word_stats: List[TranscriptionTokenStats] = []
        for (
            time_s,
            token,
            score,
        ) in zip(token_timestamps, pieces, step_scores):
            if (
                not word_stats
                or token.startswith("▁")
                and time_s > word_stats[-1].time_s
            ):
                word_stats.append(
                    TranscriptionTokenStats(
                        token.replace("▁", " ").strip(), time_s, [np.exp(score)]
                    )
                )
            else:
                word_stats[-1].text += token.replace("▁", " ")
                word_stats[-1].scores.append(np.exp(score))
        words = [
            TranscriptionToken(token.text, token.time_s, np.mean(token.scores).item())
            for token in word_stats
        ]
        return words

    def run_inference(
        self,
        fbanks: torch.Tensor,
        src_lang: str,
        length_seconds: float,
        filter_width: int,
        gen_opts: Dict,
    ) -> Transcription:
        prefix = self.tokenizer.create_encoder(
            mode="target", lang=src_lang
        ).prefix_indices
        beam_size = gen_opts.get("beam_size") or 1  # set to 1 by default
        gen_opts.pop("beam_size", None)
        generator = BeamSearchSeq2SeqGenerator(
            model=self.s2t,
            beam_size=beam_size,
            **gen_opts,
        )

        self.enc_dec_attn_collector.reset()
        assert prefix is not None
        output: Seq2SeqGeneratorOutput = generator(
            source_seqs=fbanks.unsqueeze(0),
            source_padding_mask=None,
            prompt_seqs=prefix.unsqueeze(0),
            prompt_padding_mask=None,
        )
        highest_prob_hypo = output.hypotheses[0][0]
        token_tensor = highest_prob_hypo.seq.squeeze(0)
        token_ids = token_tensor.tolist()[:-1]
        step_scores_tensor = highest_prob_hypo.step_scores
        assert step_scores_tensor is not None
        step_scores = step_scores_tensor.tolist()[:-1]
        enc_dec_attn_scores = self.enc_dec_attn_collector.attn_scores[:-1]
        token_timestamps = self._extract_timestamps(
            enc_dec_attn_scores,
            length_seconds,
            filter_width,
        )
        pieces = [
            self.tokenizer.model.index_to_token(token_id) for token_id in token_ids
        ]
        stats = self._collect_word_level_stats(
            pieces=pieces,
            token_timestamps=token_timestamps,
            step_scores=step_scores,
        )
        return Transcription(stats)

    def denoise_audio(
        self, audio: Union[str, Tensor], denoise_config: Optional[DenoisingConfig]
    ) -> AudioDecoderOutput:
        demucs = Demucs(denoise_config=denoise_config)
        audio = demucs.denoise(audio)
        assert isinstance(audio, MemoryBlock)
        return self.decode_audio(audio)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, Tensor],
        src_lang: str,
        filter_width: int = 3,
        sample_rate: int = 16000,
        denoise: bool = False,
        denoise_config: Optional[DenoisingConfig] = None,
        chunk_size_sec: int = 20,
        pause_length_sec: float = 1,
        **sequence_generator_options: Dict,
    ) -> Optional[Transcription]:
        """
        The main method used to perform transcription.

        :param audio:
            Either path to audio or audio Tensor.
        :param src_lang:
            Source language of audio.
        :param sample_rate:
            Sample rate of the audio Tensor.
        :param filter_width:
            Window size to pad weights tensor.
        :param chunk_size_sec:
            Length of audio chunks in seconds.
            For segmenting audio.
        :param pause_length_sec:
            Length of pause between audio chunks in seconds.
            For segmenting audio.
        :params **sequence_generator_options:
            See BeamSearchSeq2SeqGenerator.
        :params denoise:
            Whether to denoise the audio.
        :params denoise_config:
            Configuration for denoising.

        :returns:
            - Transcription: list of tokens with timestamps and joined text
        """

        if denoise:
            decoded_audio = self.denoise_audio(audio, denoise_config)
        else:
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

        wav = decoded_audio.get("waveform")
        assert wav is not None

        decoded_sample_rate = decoded_audio.get("sample_rate")
        assert decoded_sample_rate is not None
        assert int(decoded_sample_rate) == sample_rate

        length_seconds = wav.size(0) / sample_rate

        waveform_2d = wav
        waveform_1d = wav.view(-1)
        segmenter = SileroVADSegmenter(
            sample_rate=sample_rate,
            chunk_size_sec=chunk_size_sec,
            pause_length=pause_length_sec,
        )

        if length_seconds > chunk_size_sec:
            src_segments = segmenter.segment_long_input(waveform_1d)  # type: ignore
        else:
            src_segments = [(0, waveform_1d.size(0))]

        transcriptions: List[Transcription] = []
        for start, end in src_segments:
            segment = waveform_2d[start:end, :]
            src_segment = self.convert_to_fbank(
                {
                    "waveform": segment,
                    "sample_rate": sample_rate,
                }
            )["fbank"]
            length_seconds_segment = segment.size(0) / sample_rate
            transcription_segment = self.run_inference(
                src_segment,
                src_lang,
                length_seconds_segment,
                filter_width,
                sequence_generator_options,
            )
            transcriptions.append(transcription_segment)

        if not transcriptions:
            return None

        for idx in range(1, len(transcriptions)):
            transcriptions[0] = transcriptions[idx]

        return transcriptions[0]
