import logging
from pathlib import Path
from typing import Any, Callable, List, Union

from fairseq2.assets.card import AssetCard
from fairseq2.data import Collater
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.generation.sequence_generator import (
    Seq2SeqGenerator,
    SequenceGeneratorOptions,
    SequenceGeneratorOutput,
    VocabularyInfo,
)
from fairseq2.memory import MemoryBlock
from fairseq2.nn.transformer.multihead_attention import AttentionWeightHook
from fairseq2.typing import DataType, Device

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from seamless_communication.models.unity import (
    UnitYX2TModel,
    load_unity_model,
    load_unity_text_tokenizer,
)

_log = logging.getLogger(__name__)


class EncDecAttentionsCollect(AttentionWeightHook):
    def __init__(self):
        super().__init__()
        self.attn_scores = []

    def __call__(self, m, attn_weights) -> None:
        val = torch.clone(attn_weights).detach().sum(dim=0).squeeze(0).tolist()
        self.attn_scores.append(val)

    def reset(self):
        self.attn_scores = []


def lis(arr):
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


class Transcriber(nn.Module):
    def __init__(
        self,
        model_name_or_card: Union[str, AssetCard],
        src_lang: str,
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
            pad_idx=model.pad_idx,
        )
        self.enc_dec_attn_collector = EncDecAttentionsCollect()
        self.s2t.decoder.layers[-1].encoder_decoder_attn.register_attn_weight_hook(
            self.enc_dec_attn_collector
        )
        self.gen_opts = SequenceGeneratorOptions(beam_size=1)
        prefix = self.tokenizer.create_encoder(
            mode="target", lang=src_lang
        ).prefix_indices.tolist()
        self.prefix_len = len(prefix)
        self.generator = Seq2SeqGenerator(
            decoder=self.s2t,
            vocab_info=self.decoder_vocab_info,
            prefix_seq=torch.LongTensor(prefix, device=self.device),
            opts=self.gen_opts,
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
            pad_idx=self.tokenizer.vocab_info.pad_idx, pad_to_multiple=2
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
    def _word_stats_to_string(cls, word_stats: List[Any]) -> str:
        words = [
            f"{stat['text']}[t{stat['time_s']:0.2f}, p{stat['prob']:0.2f}]"
            for stat in word_stats
        ]
        return " ".join(words)

    @classmethod
    def _extract_timestamps(cls, attn_weights, audio_len):
        num_out_tokens = len(attn_weights)
        num_encoder_steps = len(attn_weights[0])
        attn_weights = np.array(attn_weights)
        col_maxes = np.argmax(attn_weights, axis=0)
        lis_input = [
            (out_tok_idx, -enc_bin_idx)
            for enc_bin_idx, out_tok_idx in enumerate(col_maxes)
        ]
        tok_idx_to_start_enc_bin_idx = {
            out_tok_idx: -enc_bin_idx for out_tok_idx, enc_bin_idx in lis(lis_input)[1]
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
    ):
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
            {"text": word, "time_s": start, "prob": np.mean(probs)}
            for word, start, probs in word_stats
        ]
        return word_stats

    def run_inference(self, fbanks: torch.Tensor, length_seconds: float) -> str:
        encoder_output, encoder_padding_mask = self.s2t.encode(
            fbanks.unsqueeze(0), None
        )
        self.enc_dec_attn_collector.reset()
        output: SequenceGeneratorOutput = self.generator(
            encoder_output=encoder_output, encoder_padding_mask=encoder_padding_mask
        )
        lang_token = output.results[0][0].seq.squeeze(0)[1].item()
        token_ids = output.results[0][0].seq.squeeze(0)[self.prefix_len :].tolist()
        step_scores = output.results[0][0].step_scores[self.prefix_len :].tolist()
        enc_dec_attn_scores = self.enc_dec_attn_collector.attn_scores[
            self.prefix_len - 1 :
        ]
        token_timestamps = self._extract_timestamps(enc_dec_attn_scores, length_seconds)
        pieces = [
            self.tokenizer.model.index_to_token(token_id) for token_id in token_ids
        ]
        word_stats = self._collect_word_level_stats(
            pieces=pieces, token_timestamps=token_timestamps, step_scores=step_scores
        )
        augmented_text = self._word_stats_to_string(word_stats)
        return augmented_text

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, Tensor],
        sample_rate: int = 16000,
    ) -> str:
        """
        The main method used to perform transcription.

        :param audio:
            Either path to audio or audio Tensor.
        :param src_lang:
            Source language of audio.

        :returns:
            - Transcribed text with timestamps.
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

        return self.run_inference(src, 15)  # TODO: extract length_seconds from audio
