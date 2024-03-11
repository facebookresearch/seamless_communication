# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
from torch import Tensor
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import cast, Any, Dict, List, Iterator, Optional, Union, final
from pathlib import Path

from fairseq2.assets import AssetCard
from fairseq2.assets import asset_store as default_asset_store
from fairseq2.assets import download_manager as default_download_manager
from fairseq2.nn.padding import get_seqs_and_padding_mask
from fairseq2.typing import DataType, Device
from fairseq2.gang import Gang
from fairseq2.data import (
    Collater,
    DataPipeline,
    FileMapper,
    SequenceData,
    create_bucket_sizes,
)
from fairseq2.data.text import (
    StrSplitter,
    TextTokenizer,
    read_text,
)
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter, WaveformToFbankOutput
from fairseq2.datasets.error import DatasetError
from fairseq2.datasets.loader import StandardDatasetLoader
from fairseq2.datasets.utils import all_eod

from seamless_communication.models.unity.model import UnitYBatch
from seamless_communication.models.unity import UnitTokenizer


logger = logging.getLogger(__name__)


class UnitYDataset:
    """Represents an UnitY dataset."""

    dataset_name: str
    data_dir: Path
    audio_root_dir: Path
    splits: Dict[str, Dict[str, int]]

    def __init__(
        self,
        dataset_name: str,
        data_dir: Path,
        audio_root_dir: Path,
        splits: Dict[str, Dict[str, int]],
    ) -> None:
        """
        :param dataset_name:
            The name of the dataset.
        :param data_dir:
            The directory under which the dataset files reside.
        :param audio_root_dir:
            The directory under which the audio(-zip) files reside.
        :param splits:
            The available data splits e.g. {"train": {"train_ctts": 1}}.
        """
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.audio_root_dir = audio_root_dir
        self.splits = splits

    @final
    def read(
        self,
        split: str,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: UnitTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        *,
        bucket_by_length: bool = False,
        shuffle_window_size: int = 0,
        num_prefetch: int = 0,
        num_accumulate: int = 1,
        text_prefix_lang_tok: bool = False,
        gcmvn_prosody_input: bool = True,
        gcmvn_mean: Optional[Tensor] = None,
        gcmvn_std: Optional[Tensor] = None,
    ) -> Iterator[List[UnitYBatch]]:
        pipeline = self._build_pipeline(
            split,
            text_tokenizer,
            unit_tokenizer,
            gang,
            max_seq_len,
            max_num_tokens,
            bucket_by_length,
            shuffle_window_size,
            num_prefetch,
            text_prefix_lang_tok=text_prefix_lang_tok,
            gcmvn_prosody_input=gcmvn_prosody_input,
            gcmvn_mean=gcmvn_mean,
            gcmvn_std=gcmvn_std,
        )

        while True:
            eod = False

            pipeline_iter = iter(pipeline)

            while not eod:
                batches = []

                for _ in range(num_accumulate):
                    try:
                        example = next(pipeline_iter)
                    except StopIteration:
                        break

                    batch = self._example_to_batch(example, gang.device)

                    batches.append(batch)

                eod = len(batches) != num_accumulate

                # When the pipeline is sharded, sampling and bucketing by length
                # can lead to unpredictability in the number of examples read in
                # each process. So, it is important to ensure that all processes
                # are in sync about the end of the data. If this is not the case,
                # a training loop may become stuck.
                if bucket_by_length:
                    eod = all_eod(eod, gang, logger)

                if not eod:
                    yield batches

    @staticmethod
    def _example_to_batch(example: Dict[str, Any], device: Device) -> UnitYBatch:
        sample = example["sample"]
        source_data = cast(SequenceData, sample["audio"]["data"]["fbank"])
        target_text_data = cast(SequenceData, sample["raw_tgt_text"])
        target_data = cast(SequenceData, sample["tgt_text"])
        prosody_input_seqs = None
        if (gcmvn_fbank := sample["audio"]["data"].get("gcmvn_fbank", None)) is not None:
            prosody_input_data = cast(SequenceData, gcmvn_fbank)
            prosody_input_seqs, prosody_input_padding_mask = get_seqs_and_padding_mask(
                prosody_input_data, device
            )

        source_seqs, source_padding_mask = get_seqs_and_padding_mask(
            source_data, device
        )
        target_seqs, target_padding_mask = get_seqs_and_padding_mask(
            target_data, device
        )
        target_text_seqs, target_text_padding_mask = get_seqs_and_padding_mask(
            target_text_data, device
        )

        return UnitYBatch(
            source_seqs=source_seqs,
            source_padding_mask=source_padding_mask,
            target_seqs=target_seqs,
            target_padding_mask=target_padding_mask,
            prosody_input_seqs=prosody_input_seqs,
            prosody_input_padding_mask=prosody_input_padding_mask,
            target_text_seqs=target_text_seqs,
            target_text_padding_mask=target_text_padding_mask,
            example=example,
        )

    @staticmethod
    def preprocess_unit_to_Tensor(unit_str: str) -> Tensor:
        unit_list = list(map(int, unit_str.split()))
        return torch.IntTensor(unit_list).unsqueeze(0)

    @staticmethod
    def squeeze_Tensor(tensor: Tensor) -> Tensor:
        #TODO: loose unit_tokenizer input requirement to support 1D
        return tensor[0]

    @staticmethod
    def fix_pad_0_to_1(tensor: Tensor) -> Tensor:
        tensor[tensor == 0] = 1
        return tensor

    @staticmethod
    def normalize_fbank(
        data: WaveformToFbankOutput,
        gcmvn_prosody_input: bool,
        gcmvn_mean: Tensor,
        gcmvn_std: Tensor,
    ) -> WaveformToFbankOutput:
        fbank = data["fbank"]
        std, mean = torch.std_mean(fbank, dim=0)
        data["fbank"] = fbank.subtract(mean).divide(std)
        if gcmvn_prosody_input:
            data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)

        return data

    @staticmethod
    def prepend_eos_symbol(tensor: Tensor, eos_idx: int) -> Tensor:
        tensor = torch.concat([tensor.new_full([1], eos_idx), tensor])
        return tensor

    @final
    def _build_pipeline(
        self,
        split: str,
        text_tokenizer: TextTokenizer,
        unit_tokenizer: UnitTokenizer,
        gang: Gang,
        max_seq_len: int,
        max_num_tokens: int,
        bucket_by_length: bool,
        shuffle_window_size: int,
        num_prefetch: int,
        text_prefix_lang_tok: bool = False,
        gcmvn_prosody_input: bool = True,
        gcmvn_mean: Optional[Tensor] = None,
        gcmvn_std: Optional[Tensor] = None,
    ) -> DataPipeline:
        splits = self.splits.get(split, None)

        if splits is None:
            raise ValueError(
                f"`split` must be a valid split name, but the {self.dataset_name} dataset has no split named '{split}'."
            )

        pipeline_blds = []

        decode_audio = AudioDecoder(dtype=torch.float32, device=gang.device)

        convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=False,  # standardize afterwards (to enable gcmvn)
            device=gang.device,
        )

        map_file = FileMapper(
            root_dir=self.audio_root_dir, cached_fd_count=3 * num_prefetch
        )

        # create data pipeline for each split and concat
        for dataset_name in splits:
            dataset_file = self.data_dir.joinpath(f"{dataset_name}.tsv")

            if not dataset_file.exists():
                raise DatasetError(
                    f"The dataset file '{dataset_file}' is not found in the {self.dataset_name} dataset."
                )

            with open(dataset_file, "r") as f:
                header = f.readline().strip("\n").split("\t")
                first_example = f.readline().strip("\n").split("\t")
                target_lang = first_example[header.index("tgt_lang")]

            split_tsv = StrSplitter(names=header)

            pipeline_builder = (
                read_text(dataset_file, rtrim=True, memory_map=True)
                .skip(1)
                .map(split_tsv)
            )

            if gang.size > 1:
                pipeline_builder.shard(gang.rank, gang.size)

            # source audio preprocessing
            pipeline_builder.map(
                map_file, selector="audio", num_parallel_calls=num_prefetch
            )

            pipeline_builder.map(
                [
                    decode_audio,
                    convert_to_fbank,
                    partial(
                        self.normalize_fbank,
                        gcmvn_prosody_input=gcmvn_prosody_input,
                        gcmvn_mean=gcmvn_mean,
                        gcmvn_std=gcmvn_std
                    )
                ],
                selector="audio.data",
                num_parallel_calls=num_prefetch,
            )

            # target text preprocessing
            text_token_encoder = text_tokenizer.create_encoder(
                task="translation", lang=target_lang, mode="source", device=gang.device
            )

            pipeline_builder.map(
                [
                    text_token_encoder,
                    partial(self.prepend_eos_symbol, eos_idx=text_tokenizer.vocab_info.eos_idx)
                ],
                selector="raw_tgt_text",
                num_parallel_calls=num_prefetch,
            )

            # target unit preprocessing
            unit_token_encoder = unit_tokenizer.create_encoder(
                lang=target_lang, device=gang.device
            )
            if not text_prefix_lang_tok:
                unit_token_encoder.is_nar_decoder = False

            pipeline_builder.map(
                [self.preprocess_unit_to_Tensor, unit_token_encoder, self.squeeze_Tensor],
                selector="tgt_text",
                num_parallel_calls=num_prefetch,
            )

            split_pipeline = pipeline_builder.and_return()

            # Include the dataset name and the line number with each example
            # for troubleshooting.
            split_ = DataPipeline.constant(dataset_name).and_return()

            line_nr = DataPipeline.count(
                start=gang.rank, step=gang.size
            ).and_return()

            # Zip the pipelines along with the pseudo pipelines into one
            pipeline_bld = DataPipeline.zip(
                [split_, line_nr, split_pipeline],
                ["split", "line_nr", "sample"],
            )

            pipeline_blds.append(pipeline_bld.and_return())

        concat_pipeline_blds = DataPipeline.concat(pipeline_blds)

        # Shuffle examples.
        if shuffle_window_size > 0:
            concat_pipeline_blds.shuffle(shuffle_window_size, strict=False)

        if bucket_by_length:
            # Bucket by the length of the source or target sequence. The longer
            # one will be considered the length of the example.
            bucket_sizes = create_bucket_sizes(
                max_num_tokens, max_seq_len, min_seq_len=4
            )

            concat_pipeline_blds.bucket_by_length(
                bucket_sizes,
                selector="sample.audio.data.fbank",
                skip_long_examples=True,
            )
        else:
            # TODO(balioglu): FIX!
            concat_pipeline_blds.bucket(32)

        # Collate bucketed examples into a batch (always pad fbank w/ 0)
        collater = Collater(pad_value=0)

        concat_pipeline_blds.map(collater, num_parallel_calls=num_prefetch)

        # TODO: unit_tokenizer.pad_idx is hardcoded to be 1,
        # yet, the 0 (bos_idx) is not used anywhere
        # thus, transforming 0 -> 1 is safe
        concat_pipeline_blds.map(
            self.fix_pad_0_to_1,
            selector="sample.tgt_text.seqs",
        )

        # Prefetch examples in a background thread.
        if num_prefetch > 0:
            concat_pipeline_blds.prefetch(num_prefetch)

        return concat_pipeline_blds.and_return()


def _create_unity_dataset(path: Path, card: AssetCard) -> UnitYDataset:
    # TODO: implement temperature-based sampling (currently not used)
    audio_root_dir = Path(card.field("audio_root_dir").as_(str))
    splits = card.field("splits").as_dict(Dict)
    return UnitYDataset(card.name, path, audio_root_dir, splits)


load_unity_dataset = StandardDatasetLoader[UnitYDataset](
    default_asset_store, default_download_manager, _create_unity_dataset
)
