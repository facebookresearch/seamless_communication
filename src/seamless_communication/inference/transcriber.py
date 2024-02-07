# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.generation import (
    BeamSearchSeq2SeqGenerator,
    SequenceGeneratorOutput,
)
from fairseq2.memory import MemoryBlock
from fairseq2.nn.padding import PaddingMask
from fairseq2.nn.transformer.multihead_attention import AttentionWeightHook
from fairseq2.typing import DataType, Device

import numpy as np
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torch import Tensor

from seamless_communication.models.unity import (
    UnitYX2TModel,
    load_unity_model,
    load_unity_text_tokenizer,
)


class EncDecAttentionsCollect(AttentionWeightHook):
    def __init__(self):
        super().__init__()
        self.attn_scores = []

    def __call__(self, m, attn, attn_weights) -> None:
        if attn_weights.shape[-2] > 1:
            val = torch.clone(attn_weights).detach().squeeze(0).sum(dim=0).tolist()
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


class TranscriptionToken:
    text: str
    time_s: float
    prob: float

    def __init__(self, text: str, time_s: float, prob: float):
        self.text = text
        self.time_s = time_s
        self.prob = prob


class Transcription:
    text: str
    tokens: List[TranscriptionToken]

    def __init__(self, tokens: List[TranscriptionToken]):
        self.text = " ".join([t.text for t in tokens])
        self.tokens = tokens

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
        encoder_layers: int = 6,
        decoder_layers: int = 3,
        embed_dim: int = 512,
        depthwise_conv_kernel_size: int = 31,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.embed_dim = embed_dim
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.depthwise_conv_kernel_size = depthwise_conv_kernel_size
        self.tokenizer = load_unity_text_tokenizer(model_name_or_card)
        self.decoder_vocab_info = self.tokenizer.vocab_info
        self.langs = self.tokenizer.langs

        model = self.load_model_for_inference(
            load_unity_model, model_name_or_card, device, dtype
        )
        self.s2t = UnitYX2TModel(
            encoder_frontend=model.speech_encoder_frontend,
            encoder=model.speech_encoder,
            decoder_frontend=model.text_decoder_frontend,
            decoder=model.text_decoder,
            final_proj=model.final_proj,
            target_vocab_info=self.decoder_vocab_info,
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
        return (maximum, reversed(seq))

    @staticmethod
    def generate_dtw(
        arr: np.array,
    ) -> List[Tuple[int, int]]:
        n = arr.shape[0]
        m = arr.shape[1]
        c = np.ones((n + 1, m + 1)) * np.inf
        t = np.array([[(0, 0)] * (m + 1)] * (n + 1))
        c[0, 0] = 0
        for i in range(n):
            for j in range(m):
                prev_xy = c[i, j]
                prev_x = c[i, j + 1]
                prev_y = c[i + 1, j]
                if prev_xy < prev_x and prev_xy < prev_y:
                    c[i + 1, j + 1] = arr[i, j] + prev_xy
                    t[i + 1, j + 1] = (1, 1)
                elif prev_x < prev_xy and prev_x < prev_y:
                    c[i + 1, j + 1] = arr[i, j] + prev_x
                    t[i + 1, j + 1] = (1, 0)
                else:
                    c[i + 1, j + 1] = arr[i, j] + prev_y
                    t[i + 1, j + 1] = (0, 1)

        path = []
        i, j = n, m
        t[0, :] = (0, 1)
        t[:, 0] = (1, 0)
        while i > 0 or j > 0:
            path.append((i - 1, j - 1))
            i -= t[i, j][0]
            j -= t[i, j][1]

        return path[::-1]

    @classmethod
    def _extract_timestamps(
        cls,
        attn_weights,
        n_scores,
        audio_len,
        filter_width,
        filter_type,
        use_dtw,
    ) -> List[float]:
        attn_weights = attn_weights[:n_scores]  # matching lengths

        num_out_tokens = len(attn_weights)
        num_encoder_steps = len(attn_weights[0])
        attn_weights = np.array(attn_weights)
        attn_weights = attn_weights / attn_weights.sum(axis=0, keepdims=1)  # normalize
        if filter_width > 0:
            if filter_type == "median":
                attn_weights = medfilt2d(attn_weights, kernel_size=(1, filter_width))
            elif filter_type == "gaussian":
                attn_weights = gaussian_filter(
                    attn_weights, sigma=1, axes=(1), radius=filter_width // 2
                )
        if not use_dtw:  # longest increasing subsequence
            col_maxes = np.argmax(attn_weights, axis=0)
            lis_input = [
                (out_tok_idx, -enc_bin_idx)
                for enc_bin_idx, out_tok_idx in enumerate(col_maxes)
            ]
            tok_idx_to_start_enc_bin_idx = {
                out_tok_idx: -enc_bin_idx
                for out_tok_idx, enc_bin_idx in cls.generate_lis(lis_input)[1]
            }
        else:  # dynamic time warping
            dtw_path = cls.generate_dtw(-attn_weights)
            tok_idx_to_start_enc_bin_idx = {
                out_tok_idx: enc_bin_idx
                for out_tok_idx, enc_bin_idx in reversed(dtw_path)
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
        word_stats: List[List[Any]] = []
        for (
            time_s,
            token,
            score,
        ) in zip(token_timestamps, pieces, step_scores):
            if not word_stats or token.startswith("▁") and time_s > word_stats[-1][1]:
                word_stats.append(
                    [token.replace("▁", " ").strip(), time_s, [np.exp(score)]]
                )
            else:
                word_stats[-1][0] += token.replace("▁", " ")
                word_stats[-1][2].append(np.exp(score))
        word_stats = [
            TranscriptionToken(word, start, np.mean(probs))
            for word, start, probs in word_stats
        ]
        return word_stats

    def run_inference(
        self,
        fbanks: torch.Tensor,
        src_lang: str,
        length_seconds: float,
        filter_width: int,
        filter_type: str,
        use_dtw: bool,
        rerun_decoder: bool,
        gen_opts: Dict,
    ) -> Transcription:
        prefix = self.tokenizer.create_encoder(
            mode="target", lang=src_lang
        ).prefix_indices
        prefix_len = len(prefix.tolist())
        generator = BeamSearchSeq2SeqGenerator(
            model=self.s2t,
            **gen_opts,
        )

        self.enc_dec_attn_collector.reset()
        output: SequenceGeneratorOutput = generator(
            source_seqs=fbanks.unsqueeze(0),
            source_padding_mask=None,
            prompt_seqs=prefix.unsqueeze(0),
            prompt_padding_mask=None,
        )

        if rerun_decoder:
            self.enc_dec_attn_collector.reset()

            tokens = output.hypotheses[0][0].seq
            tokens_padding_mask = PaddingMask(
                seq_lens=torch.LongTensor([tokens.shape[-1]]),
                batch_seq_len=tokens.shape[-1],
            )
            seqs, padding_mask = self.s2t.decoder_frontend(
                seqs=tokens.unsqueeze(0),
                padding_mask=tokens_padding_mask,
            )
            self.s2t.decoder(
                seqs=seqs,
                padding_mask=padding_mask,
                encoder_output=output.encoder_output,
                encoder_padding_mask=output.encoder_padding_mask,
            )

        token_ids = output.hypotheses[0][0].seq.squeeze(0)[prefix_len:].tolist()
        step_scores = output.hypotheses[0][0].step_scores[prefix_len:].tolist()
        enc_dec_attn_scores = self.enc_dec_attn_collector.attn_scores[prefix_len - 1 :]
        token_timestamps = self._extract_timestamps(
            enc_dec_attn_scores,
            len(step_scores),
            length_seconds,
            filter_width,
            filter_type,
            use_dtw,
        )
        pieces = [
            self.tokenizer.model.index_to_token(token_id) for token_id in token_ids
        ]
        stats = self._collect_word_level_stats(
            pieces=pieces, token_timestamps=token_timestamps, step_scores=step_scores
        )
        return Transcription(stats)

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, Tensor],
        src_lang: str,
        filter_width: int = 0,
        filter_type: str = "",
        sample_rate: int = 16000,
        use_dtw: bool = False,
        rerun_decoder: bool = True,
        **sequence_generator_options: Dict,
    ) -> Transcription:
        """
        The main method used to perform transcription.

        :param audio:
            Either path to audio or audio Tensor.
        :param src_lang:
            Source language of audio.
        :param sample_rate:
            Sample rate of the audio Tensor.
        :param use_dtw:
            Use Dynamic Time Warping to extract timestamps
            rather than default Longest Increasing Subsequence
        :param filter_width:
            Window size to padding weights tensor.
        :param filter_type:
            Filter algorithm to pad weights tensor (`""`, `"median"`, `"gaussian"`).
        :params **sequence_generator_options:
            See BeamSearchSeq2SeqGenerator.

        :returns:
            - List of Tokens with timestamps.
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

        src = self.convert_to_fbank(decoded_audio)["fbank"]

        length_seconds = (
            decoded_audio["waveform"].size(0) / decoded_audio["sample_rate"]
        )

        return self.run_inference(
            src,
            src_lang,
            length_seconds,
            filter_width,
            filter_type,
            use_dtw,
            rerun_decoder,
            sequence_generator_options,
        )
