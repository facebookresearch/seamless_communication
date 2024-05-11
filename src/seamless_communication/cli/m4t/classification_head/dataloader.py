# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.


import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
import torchaudio
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from fairseq2.data.text import TextTokenEncoder
from fairseq2.models.nllb import NllbTokenizer
from fairseq2.data.audio import WaveformToFbankConverter
from torch import Tensor
from torch.nn.functional import pad as pad_tensor
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from seamless_communication.datasets.datatypes import LangPairSample
from seamless_communication.models.unity.unit_tokenizer import (
    UnitTokenEncoder,
    UnitTokenizer,
)

logger = logging.getLogger(__name__)


@dataclass
class SeqsBatch:
    src_tokens: Optional[Tensor]
    src_lengths: Optional[Tensor]

    def __del__(self) -> None:
        """Explicitly delete tensors
        to force GPU memory cleanup"""
        for tensor in [
            self.src_tokens,
            self.src_lengths
        ]:
            if tensor is not None:
                del tensor


@dataclass
class BatchingConfig:
    fbank_feats_pad_idx: int = 0
    """The pad index to use in fbanks batching."""

    batch_size: int = 5
    """Fixed batch size to use"""

    max_audio_length_sec: float = 15.0
    """ Drop samples with source audio sample length above the threshold."""

    rank: int = 0
    """The rank of this worker in the process group."""

    world_size: int = 1
    """The world size of the process group."""

    num_workers: int = 2
    """Parallelism in dataset preparation."""

    float_dtype: torch.dtype = torch.float16
    """Select between fp16/fp32 for float tensors """


def worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)  # type: ignore


class UnitYLanguageIDDataLoader:
    SAMPLE_RATE = 16_000

    def __init__(
        self,
        num_languages: int,
        text_tokenizer: NllbTokenizer,
        unit_tokenizer: UnitTokenizer,
        dataset_manifest_path: str,
        batching_config: BatchingConfig,
    ):
        self.num_languages = num_languages
        self.text_tokenizer = text_tokenizer
        self.text_encoders_per_lang: Dict[str, TextTokenEncoder] = {}
        self.unit_tokenizer = unit_tokenizer
        self.unit_encoders_per_lang: Dict[str, UnitTokenEncoder] = {}
        self.batching_config = batching_config
        self._fbank_extract_params = {
            "num_mel_bins": 80,
            "waveform_scale": 32768,
            "channel_last": True,
            "standardize": True,
            "device": torch.device("cpu"),
            "dtype": self.batching_config.float_dtype,
        }
        self.dataset = self._load_manifest(dataset_manifest_path)

    def get_dataloader(self) -> DataLoader[SeqsBatch]:
        subset = split_dataset_by_node(
            self.dataset,
            rank=self.batching_config.rank,
            world_size=self.batching_config.world_size,
        )
        data_loader = DataLoader(
            dataset=subset,
            batch_size=self.batching_config.batch_size,
            shuffle=True,
            num_workers=self.batching_config.num_workers,
            collate_fn=self._collate,
            worker_init_fn=worker_init_fn,
        )
        return data_loader

    def __iter__(self) -> Iterable:
        return self.get_dataloader().__iter__()

    def _get_source_fbank(self, sample: LangPairSample) -> Tensor:
        wav, sample_rate = torchaudio.load(sample.source.audio_local_path)
        assert (
            int(sample_rate) == self.SAMPLE_RATE
        ), f"sample != {self.SAMPLE_RATE}, please resample"
        assert len(wav.shape) in (1, 2)
        if len(wav.shape) == 1:
            wav = wav.unsqueeze(-1)
        elif wav.shape[0] <= 2:  # channel is first, should be second
            wav = wav.transpose(0, 1)
        return WaveformToFbankConverter(**self._fbank_extract_params)(  # type: ignore
            {
                "waveform": wav,
                "sample_rate": self.SAMPLE_RATE,
            }
        )["fbank"]

    def _batch_tensors(self, tensors: List[Tensor], pad_value: Any) -> Tensor:
        padding_size = max(tensor.shape[0] for tensor in tensors)
        dims = len(tensors[0].shape)
        padded_tensors = []
        for tensor in tensors:
            padding = [0] * 2 * dims
            padding[-1] = padding_size - tensor.shape[0]
            padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
        return torch.stack([tensor for tensor in padded_tensors], dim=0)

    def _is_long_src_audio(self, sample: LangPairSample) -> bool:
        # HACK:: causes errored audios to be excluded but this is difficult to follow
        try:
            wav, sample_rate = torchaudio.load(sample.source.audio_local_path)
            length_s: float = max(wav.shape) / sample_rate
            return length_s > self.batching_config.max_audio_length_sec
        except:
            logger.exception(f"Failed to load sample path: {sample.source.audio_local_path}")
            return True

    def _collate(self, raw_samples: List[Dict[str, Any]]):
        samples = [ LangPairSample.from_json(sample) for sample in raw_samples ]
        
        ## Input Speech
        
        # 1 - filter long audio samples
        filtered_samples = [ sample for sample in samples if not self._is_long_src_audio(sample) ]
        samples = filtered_samples if filtered_samples else [samples[0]]  # keep at least one sample
        src_tokens_list = [ self._get_source_fbank(sample) for sample in samples ]
        
        # 2 - filter NaNs in fbanks´´
        with_nans = [ fbank.isnan().any().item() for fbank in src_tokens_list ]
        samples = [ sample for sample, skip in zip(samples, with_nans) if not skip ]
        assert len(samples) > 0
        src_tokens_list = [ tok for tok, skip in zip(src_tokens_list, with_nans) if not skip ]
        src_tokens = self._batch_tensors(
            src_tokens_list, pad_value=self.batching_config.fbank_feats_pad_idx
        ).to(self.batching_config.float_dtype)
        src_lengths = torch.LongTensor([ tok.shape[0] for tok in src_tokens_list ])
        
        ## Output Label
        le = LabelEncoder()
        source_langs = [ sample.source.lang for sample in samples ]
        onehot_labels = torch.nn.functional.one_hot(
            torch.tensor(le.fit_transform(source_langs)),
            num_classes=self.num_languages).float()
        
        while src_tokens.size(0) < self.batching_config.batch_size:
            src_tokens = torch.cat((src_tokens, src_tokens[-1].unsqueeze(0)), dim=0)
            src_lengths = torch.cat((src_lengths, src_lengths[-1].unsqueeze(0)), dim=0)
            onehot_labels = torch.cat((onehot_labels, onehot_labels[-1].unsqueeze(0)), dim=0)
            
        return SeqsBatch(src_tokens=src_tokens, src_lengths=src_lengths), onehot_labels

    def _load_manifest(self, dataset_manifest_path: str) -> Dataset:
        with open(dataset_manifest_path) as fp_in:
            dataset = [json.loads(line) for line in fp_in]
            return Dataset.from_list(dataset)
