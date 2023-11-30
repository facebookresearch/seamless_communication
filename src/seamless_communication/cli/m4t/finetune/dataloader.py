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
import torchaudio.compliance.kaldi as ta_kaldi
from datasets import Dataset
from datasets.distributed import split_dataset_by_node
from fairseq2.data.text import TextTokenEncoder
from fairseq2.models.nllb import NllbTokenizer
from torch import Tensor
from torch.nn.functional import pad as pad_tensor
from torch.utils.data import DataLoader

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
    target_tokens: Optional[Tensor]
    prev_output_tokens: Optional[Tensor]
    target_lengths: Optional[Tensor]

    def __del__(self) -> None:
        """Explicitly delete tensors
        to force GPU memory cleanup"""
        for tensor in [
            self.src_tokens,
            self.src_lengths,
            self.target_tokens,
            self.prev_output_tokens,
            self.target_lengths,
        ]:
            if tensor is not None:
                del tensor


@dataclass
class MultimodalSeqsBatch:
    speech_to_text: SeqsBatch
    text_to_units: SeqsBatch

    def __del__(self) -> None:
        del self.speech_to_text
        del self.text_to_units


@dataclass
class BatchingConfig:
    fbank_feats_pad_idx: int = 0
    """The pad index to use in fbanks batching."""

    batch_size: int = 5

    rank: int = 0
    """The rank of this worker in the process group."""

    world_size: int = 1
    """The world size of the process group."""

    num_workers: int = 2
    """Parallelism in dataset preparation."""

    float_dtype: torch.dtype = torch.float16
    """Select between fp16/fp32 for float tensors """


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class UnitYDataLoader:
    def __init__(
        self,
        text_tokenizer: NllbTokenizer,
        unit_tokenizer: UnitTokenizer,
        dataset_manifest_path: str,
        batching_config: BatchingConfig,
    ):
        self.text_tokenizer = text_tokenizer
        self.text_encoders_per_lang: Dict[str, TextTokenEncoder] = {}
        self.unit_tokenizer = unit_tokenizer
        self.unit_encoders_per_lang: Dict[str, UnitTokenEncoder] = {}
        self.batching_config = batching_config
        self.dataset = self._load_manifest(dataset_manifest_path)

    def get_dataloader(self) -> DataLoader:
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
            collate_fn=self._prepare_batch,
            worker_init_fn=worker_init_fn,
        )
        return data_loader

    def __iter__(self) -> Iterable[MultimodalSeqsBatch]:
        return self.get_dataloader().__iter__()

    def _get_source_fbank(self, sample: LangPairSample) -> Tensor:
        audio_input = torchaudio.load(sample.source.audio_local_path)[0]
        return ta_kaldi.fbank(audio_input, num_mel_bins=80)

    def _get_tokenized_target_text(self, sample: LangPairSample) -> Tensor:
        """Expected sequence is [<eos>, <lang_tok> , ..text tokens.., <eos>]"""
        target_lang = sample.target.lang
        if target_lang not in self.text_encoders_per_lang:
            self.text_encoders_per_lang[
                target_lang
            ] = self.text_tokenizer.create_encoder(lang=target_lang, mode="target")
        tokens = self.text_encoders_per_lang[target_lang](sample.target.text)
        eos_idx = self.text_tokenizer.vocab_info.eos_idx
        tokens = torch.concat([tokens, torch.LongTensor([eos_idx])])
        return tokens

    def _get_tokenized_units(self, sample: LangPairSample) -> Optional[Tensor]:
        """Expected sequence is [<eos>, <lang_tok> , ..unit tokens.., <eos>]"""
        if sample.target.units is None:
            return None
        target_lang = sample.target.lang
        if target_lang not in self.unit_encoders_per_lang:
            self.unit_encoders_per_lang[
                target_lang
            ] = self.unit_tokenizer.create_encoder(lang=target_lang)
        tokens = self.unit_encoders_per_lang[target_lang](
            torch.LongTensor(sample.target.units).unsqueeze(0)
        )
        eos_idx = self.unit_tokenizer.vocab_info.eos_idx
        tokens = torch.concat([tokens.squeeze(0), torch.LongTensor([eos_idx])])
        return tokens

    def _batch_tensors(self, tensors: List[Tensor], pad_value: Any) -> Tensor:
        padding_size = max(tensor.shape[0] for tensor in tensors)
        dims = len(tensors[0].shape)
        padded_tensors = []
        for tensor in tensors:
            padding = [0] * 2 * dims
            padding[-1] = padding_size - tensor.shape[0]
            padded_tensors.append(pad_tensor(tensor, padding, "constant", pad_value))
        return torch.stack([tensor for tensor in padded_tensors], dim=0)

    def _prepare_batch(self, raw_samples: List[Dict[str, Any]]) -> MultimodalSeqsBatch:
        samples = [LangPairSample.from_json(sample) for sample in raw_samples]
        # input speech
        src_tokens_list = [self._get_source_fbank(sample) for sample in samples]
        src_tokens = self._batch_tensors(
            src_tokens_list, pad_value=self.batching_config.fbank_feats_pad_idx
        ).to(self.batching_config.float_dtype)
        src_lengths = torch.LongTensor(
            [src_tokens.shape[0] for src_tokens in src_tokens_list]
        )
        # output text
        text_tokens_list = [
            self._get_tokenized_target_text(sample) for sample in samples
        ]
        text_pad_idx = self.text_tokenizer.vocab_info.pad_idx
        prev_outputs_tokens = self._batch_tensors(
            [tokens[:-1] for tokens in text_tokens_list], pad_value=text_pad_idx
        )
        target_tokens = self._batch_tensors(
            [tokens[1:] for tokens in text_tokens_list], pad_value=text_pad_idx
        )
        tokens_lengths = torch.LongTensor(
            [tokens.shape[0] - 1 for tokens in text_tokens_list]
        )
        # output units
        units_list_raw = [self._get_tokenized_units(sample) for sample in samples]
        if None in units_list_raw:
            prev_outputs_units = None
            target_units = None
            units_lengths = None
        else:
            units_list: List[Tensor] = [
                value for value in units_list_raw if value is not None
            ]
            units_pad_idx = self.unit_tokenizer.vocab_info.pad_idx
            prev_outputs_units = self._batch_tensors(
                [tokens[:-1] for tokens in units_list], pad_value=units_pad_idx
            )
            target_units = self._batch_tensors(
                [tokens[1:] for tokens in units_list], pad_value=units_pad_idx
            )
            units_lengths = torch.LongTensor(
                [tokens.shape[0] - 1 for tokens in units_list]
            )
        return MultimodalSeqsBatch(
            speech_to_text=SeqsBatch(
                src_tokens=src_tokens,
                src_lengths=src_lengths,
                target_tokens=target_tokens,
                prev_output_tokens=prev_outputs_tokens,
                target_lengths=tokens_lengths,
            ),
            text_to_units=SeqsBatch(
                src_tokens=None,
                src_lengths=None,
                target_tokens=target_units,
                prev_output_tokens=prev_outputs_units,
                target_lengths=units_lengths,
            ),
        )

    def _load_manifest(self, dataset_manifest_path: str) -> Dataset:
        with open(dataset_manifest_path) as fp_in:
            dataset = [json.loads(line) for line in fp_in]
            return Dataset.from_list(dataset)
