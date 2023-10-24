# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple, Union
import ctypes

import torch
from m4t_scripts.train.configs import AudioProcessingConfig, DataLoadingConfig
from torch import Tensor

from fairseq2.data import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    FileMapper,
)
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.text import SentencePieceEncoder, StrSplitter, read_text
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from seamless_communication.models.tokenizer import SPMTokenizer
from seamless_communication.models.unity import (
    UnitTokenizer,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

logger = logging.getLogger(__name__)


class SeqsBatch(NamedTuple):
    src_tokens: Optional[Tensor]
    src_lengths: Optional[Tensor]
    target_tokens: Tensor
    prev_output_tokens: Tensor
    target_lengths: Tensor
    prefix_tokens: Optional[Tensor]


class MultimodalSeqsBatch(NamedTuple):
    speech_to_text: SeqsBatch
    text_to_units: SeqsBatch


class UnityDataLoader:
    CPU_DEVICE = torch.device("cpu")
    MANIFEST_EXT = ".tsv"
    MANIFEST_COLUMN_SEP = "\t"
    AUDIO_COLUMN_NAME = "audio"
    TARGET_TEXT_COLUMN = "raw_tgt_text"
    TARGET_UNITS_COLUMN = "tgt_text"
    TARGET_LANG_COLUMN = "tgt_lang"
    ROOT_COLUMN = "_"
    BATCH_WIDTH_STEP = 8

    def __init__(
        self,
        config: DataLoadingConfig,
        rank: int = 0,
        world_size: int = 1,
        target_device: torch.device = CPU_DEVICE,
        float_dtype: torch.dtype = torch.float16,  # training/inference precision
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.target_device = target_device
        self.float_dtype = float_dtype
        self._set_mkl_num_threads()
        self.manifest_paths = list(self._iterate_manifest_paths())
        self.text_tokenizer = self._init_text_tokenizer()
        self.unit_tokenizer = self._init_unit_tokenizer()
        self.spm_encoder = SentencePieceEncoder(
            model=self.text_tokenizer.model, suffix_tokens=["</s>"]
        )
        self.text_prefix_tokens = self._build_text_tgt_prefixes()
        self.unit_prefix_tokens = self._build_unit_tgt_prefixes()
        if self.config.fixed_batch_size is None:
            self.tgt_text_batch_shapes = self._calculate_tgt_text_batch_shapes()
        else:
            self.tgt_text_batch_shapes = []

        self.pipeline = self._build_pipeline()

    @classmethod
    def _set_mkl_num_threads(cls):
        """Setting mkl num threads to 1, so that we don't get thread explosion."""
        mkl_rt = ctypes.CDLL("libmkl_rt.so")
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))

    def _calculate_tgt_text_batch_shapes(self) -> List[Tuple[int, int]]:
        max_seq_len = self.config.max_tgt_text_tokens_per_sample
        max_tokens_per_batch = self.config.max_tgt_text_tokens_per_batch
        assert max_tokens_per_batch is not None, "max_tokens_per_batch is not set"
        max_bsz = (
            self.config.max_batch_size
            if self.config.max_batch_size is not None
            else max_tokens_per_batch
        )
        step = self.BATCH_WIDTH_STEP
        bucket_sizes = []
        for seq_len in range(step, max(step, max_seq_len) + 1, step):
            bsz = max(1, max_tokens_per_batch // seq_len)
            if bsz > max_bsz:
                continue
            bucket_sizes.append((bsz, seq_len))
        return bucket_sizes

    def _build_text_tgt_prefixes(self) -> Dict[str, List[int]]:
        return {
            lang_tok: self.text_tokenizer.create_encoder(
                lang=lang_tok, mode="target"
            ).prefix_indices.tolist()  # type:ignore
            for lang_tok in self.text_tokenizer.langs
        }

    def _build_unit_tgt_prefixes(self) -> Dict[str, List[int]]:
        assert self.unit_tokenizer.vocab_info.eos_idx is not None
        return {
            lang_tok: [
                self.unit_tokenizer.vocab_info.eos_idx,
                self.unit_tokenizer.lang_to_index(lang_tok),
            ]
            for lang_tok in self.unit_tokenizer.langs
        }  # type: ignore

    def _init_text_tokenizer(self) -> Union[NllbTokenizer, SPMTokenizer]:
        if self.config.text_tokenization.from_model is not None:
            return load_unity_text_tokenizer(self.config.text_tokenization.from_model)
        else:
            assert self.config.text_tokenization.langtoks is not None
            assert self.config.text_tokenization.spm_path is not None
            return SPMTokenizer(
                pathname=self.config.text_tokenization.spm_path,
                langs=self.config.text_tokenization.langtoks,
            )

    def _init_unit_tokenizer(self) -> UnitTokenizer:
        if self.config.unit_tokenization.from_model is not None:
            return load_unity_unit_tokenizer(self.config.unit_tokenization.from_model)
        else:
            raise NotImplementedError("TBD")

    def _load_manifest_list_from_file(self) -> Iterator[str]:
        if self.config.manifest_list_path is not None:
            for line in open(self.config.manifest_list_path).readlines():
                line = line.split("#")[0].strip()  # allow comments
                if line:
                    yield line

    def _load_raw_manifest_list(self) -> List[str]:
        raw_list = []
        if self.config.manifest_list is not None:
            raw_list += self.config.manifest_list.strip().split(",")
        raw_list += list(self._load_manifest_list_from_file())
        return raw_list

    def _infer_manifest_full_path(self, manifest_name: str) -> str:
        full_path = manifest_name.strip()
        if self.config.manifest_path_prefix is not None:
            full_path = os.path.join(
                self.config.manifest_path_prefix.strip(), full_path
            )
        if not full_path.endswith(self.MANIFEST_EXT) and not os.path.exists(full_path):
            full_path += self.MANIFEST_EXT
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found {full_path}")
        return full_path

    def _iterate_manifest_paths(self, skip_missing_files: bool = True) -> Iterator[str]:
        """Yields full paths to manifests described in the data config.
        Check that each file exist.
        Expects *.tsv files"""
        raw_list = self._load_raw_manifest_list()
        for manifest_name in raw_list:
            try:
                full_path = self._infer_manifest_full_path(manifest_name=manifest_name)
            except FileNotFoundError:
                if skip_missing_files:
                    logger.warning(f"Skipping manifest {manifest_name}, file not found")
                    continue
                raise
            yield full_path

    def _read_column_names(self, manifest_path: str) -> List[str]:
        """Gets the order of columns in the manifest file.
        Also checks that expected columns are present."""
        with open(manifest_path, "r") as in_fp:
            column_names = in_fp.readline().strip().split("\t")
        for column in [
            self.AUDIO_COLUMN_NAME,
            self.TARGET_TEXT_COLUMN,
            self.TARGET_UNITS_COLUMN,
            self.TARGET_LANG_COLUMN,
        ]:
            if column not in column_names:
                raise ValueError(
                    f"Column `{column}` is not present in `{manifest_path}` "
                )
        return column_names

    def _builder_from_manifest(self, manifest_path: str) -> DataPipelineBuilder:
        """Creates a data pipeline builder for the specified manifest_path file."""
        logger.debug(f"Initialiazing samples loader from {manifest_path}")

        # Memory map file and read it in text mode (skip empty lines if any).
        # Skip header.
        tsv_lines = (
            read_text(
                pathname=manifest_path,
                encoding="UTF-8",
                rtrim=True,
                skip_empty=True,
                memory_map=True,
            )
            .skip(1)
            .and_return()
        )

        # Assing column names:
        # line content: `_`
        # source manifest path: `manifest_path`
        # line number: `lineno`
        line_numbers = DataPipeline.count().and_return()
        filename_const = DataPipeline.constant(manifest_path).and_return()
        pipeline = DataPipeline.zip(
            [tsv_lines, filename_const, line_numbers],
            names=[self.ROOT_COLUMN, "manifest_path", "lineno"],
            zip_to_shortest=True,
        )

        # Read every `world_size`th line starting from `rank`th item in the file.
        pipeline.shard(self.rank, self.world_size)

        if self.config.shuffle_window is not None:
            pipeline.shuffle(self.config.shuffle_window)

        # Split each text line into its fields.
        fields = self._read_column_names(manifest_path)
        logger.debug(f"Column names: {fields}")
        txt_splitter = StrSplitter(
            sep=self.MANIFEST_COLUMN_SEP, names=fields, indices=[], exclude=True
        )
        pipeline.map(
            txt_splitter,
            selector=self.ROOT_COLUMN,
            num_parallel_calls=self.config.num_threads,
        )
        # And, create the pipeline for the TSV file.
        return pipeline

    def _get_manifest_funnel(self) -> DataPipelineBuilder:
        """Creates a joined pipeline from all manifests.
        Picks samples from per-manifest pipelines in a round-robin order"""
        # TODO: add the ability to upsample/downsample manifests
        logger.info(f"Aggregating data from {len(self.manifest_paths)} manifests")
        builders = [
            self._builder_from_manifest(manifest_path=path)
            for path in self.manifest_paths
        ]
        pipelines = [builder.and_return() for builder in builders]
        return DataPipeline.round_robin(pipelines=pipelines)

    def _attach_audio(self, builder: DataPipelineBuilder) -> DataPipelineBuilder:
        """Attaches audio waveforms and fbanks from linked autio files"""
        audio_selector = f"{self.ROOT_COLUMN}.{self.AUDIO_COLUMN_NAME}"
        audio_data_selector = f"{audio_selector}.data"

        # Memory map each `audio_file`
        map_file = FileMapper(self.config.audio.audio_root_dir, cached_fd_count=100)
        builder.map(
            map_file,
            selector=audio_selector,
            num_parallel_calls=self.config.num_threads,
        )

        # Decode each mmap'ed audio file using libsndfile.
        decode_audio = AudioDecoder(dtype=torch.float32)
        builder.map(
            decode_audio,
            selector=audio_data_selector,
            num_parallel_calls=self.config.num_threads,
        )

        # And, convert from waveform to log-mel filterbank
        convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=self.config.audio.fbanks_num_mel_bins,
            waveform_scale=self.config.audio.fbanks_waveform_scale,
            channel_last=True,  # audio channel is the last dimension in the waveform
            standardize=self.config.audio.fbanks_standardize_audio,
            keep_waveform=False,
            device=self.CPU_DEVICE,  # avoid uncontrolled memory cons on GPUs
            dtype=self.float_dtype,
        )
        builder.map(
            convert_to_fbank,
            selector=audio_data_selector,
            num_parallel_calls=self.config.num_threads,
        )
        return builder

    def _attach_target_tokens(
        self, builder: DataPipelineBuilder
    ) -> DataPipelineBuilder:
        # Convert `raw_tgt_text` to (full) target tokenized sequences:
        #                   <eos> <lang_tok> <tokens .. > <eos>
        # Lang tokens change between rows, so can't use static encoder
        builder.map(
            [self.spm_encoder],
            selector=f"{self.ROOT_COLUMN}.{self.TARGET_TEXT_COLUMN}",
            num_parallel_calls=self.config.num_threads,
        )

        # Convert the `tgt_text` field into a unit tensor + EOS
        # TODO: We should use unit tokenizer.
        # Motivation for the current implementation:
        # 1) lang_tok can change between rows.
        #       If we want to attach lang_token_id here, we need a way to join values from two columns
        # 2) StrToTensorConverter doesn't allow suffix tokens. Adding it later is less covenient.
        # 3) Not a computational blocker
        convert_to_units = lambda units_str: (  # noqa: E731
            torch.LongTensor(
                [
                    int(unit_id) + 4
                    for unit_id in units_str.rstrip().bytes().decode("utf-8").split()
                ]
                + [self.unit_tokenizer.vocab_info.eos_idx]
            )
        )
        builder.map(
            [convert_to_units],
            selector=f"{self.ROOT_COLUMN}.{self.TARGET_UNITS_COLUMN}",
            num_parallel_calls=self.config.num_threads,
        )

        # prefixes for tokenized texts and speech units (<eos> <lang_tok>)
        prefix_builder = lambda lang_tok: torch.LongTensor(  # noqa: E731
            [
                self.text_prefix_tokens[lang_tok.bytes().decode("utf8")],
                self.unit_prefix_tokens[lang_tok.bytes().decode("utf8")],
            ]
        )
        builder.map(
            [prefix_builder],
            selector=f"{self.ROOT_COLUMN}.{self.TARGET_LANG_COLUMN}",
            num_parallel_calls=self.config.num_threads,
        )
        return builder

    def _get_input_audio_seconds(self, sample: Any) -> float:
        audio_data = sample[self.ROOT_COLUMN][self.AUDIO_COLUMN_NAME]["data"]
        input_audio_sample_rate = audio_data["sample_rate"]
        num_fbanks = max(audio_data["fbank"].shape)  # not guessing the dim order
        # TODO: clarify where '* 2' comes from
        waveform_length = num_fbanks * self.config.audio.fbanks_num_mel_bins * 2
        input_audio_seconds = waveform_length / input_audio_sample_rate
        return input_audio_seconds

    def _is_long_sample(self, sample: Any) -> bool:
        # input audio length
        if (
            self._get_input_audio_seconds(sample)
            > self.config.max_seconds_per_input_audio
        ):
            return True

        # target text tokens
        num_tgt_text_tokens = sample[self.ROOT_COLUMN][self.TARGET_TEXT_COLUMN].shape[
            -1
        ]
        if num_tgt_text_tokens > self.config.max_tgt_text_tokens_per_sample:
            return True

        # target units
        num_tgt_units = sample[self.ROOT_COLUMN][self.TARGET_UNITS_COLUMN].shape[
            -1
        ]  # target units
        if num_tgt_units > self.config.max_units_per_sample:
            return True
        return False

    def _nans_in_fbanks(self, sample: Any) -> bool:
        """Tells if NaNs present in fbank"""
        fbank = sample[self.ROOT_COLUMN][self.AUDIO_COLUMN_NAME]["data"]["fbank"]
        has_nans: bool = torch.any(torch.isnan(fbank)).item()  # type: ignore
        if has_nans:
            logger.warning("Sample fbank contains NaNs. Skipping")
        return has_nans

    def _filter_samples(self, builder: DataPipelineBuilder) -> DataPipelineBuilder:
        # Drop:
        #  - "long" samples
        #  - samples with fbanks that contain NaNs
        builder.filter(
            lambda sample: not self._is_long_sample(sample)
            and not self._nans_in_fbanks(sample)
        )
        return builder

    def _batch_samples(self, builder: DataPipelineBuilder) -> DataPipelineBuilder:
        if self.config.fixed_batch_size is not None:
            builder.bucket(bucket_size=self.config.fixed_batch_size)
        elif self.tgt_text_batch_shapes is not None:
            builder.bucket_by_length(
                self.tgt_text_batch_shapes,
                selector=f"{self.ROOT_COLUMN}.{self.TARGET_TEXT_COLUMN}",
            )
        else:
            raise ValueError("Unclear batching strategy")
        # Collate bucketed elements into a batch.
        collater = Collater(
            pad_to_multiple=1,
            overrides=[
                CollateOptionsOverride(
                    selector=f"{self.ROOT_COLUMN}.{self.AUDIO_COLUMN_NAME}.data.fbank",
                    pad_idx=self.config.fbank_feats_pad_idx,
                ),
                CollateOptionsOverride(
                    selector=f"{self.ROOT_COLUMN}.{self.TARGET_TEXT_COLUMN}",
                    pad_idx=self.text_tokenizer.vocab_info.pad_idx,
                ),
                CollateOptionsOverride(
                    selector=f"{self.ROOT_COLUMN}.{self.TARGET_UNITS_COLUMN}",
                    pad_idx=self.unit_tokenizer.vocab_info.pad_idx,
                ),
            ],
        )
        builder.map(collater, num_parallel_calls=self.config.num_threads)
        if self.config.prefech_batches is not None:
            builder.prefetch(self.config.prefech_batches)
        return builder

    def _build_pipeline(self) -> DataPipeline:
        data = self._get_manifest_funnel()
        data = self._attach_audio(data)
        data = self._attach_target_tokens(data)
        data = self._filter_samples(data)
        batches = self._batch_samples(data)
        return batches.and_return()

    def _gen_prev_toks_target_toks_target_lens(
        self, seqs: Any, prefix_tokens: torch.Tensor, pad_idx: int, eos_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # <eos> <lang_tok> ... <eos> <pad>*
        tokens = torch.cat((prefix_tokens, seqs["seqs"]), 1)
        target_lengths = seqs["seq_lens"] + 1  # + <leng_tok>

        prev_output_tokens = torch.clone(tokens)
        # replace last <eos> with <pad> and remove last column
        mask = prev_output_tokens == eos_idx
        mask[:, 0] = 0
        prev_output_tokens[mask] = pad_idx
        prev_output_tokens = prev_output_tokens[:, :-1]

        target_tokens = tokens[:, 1:]
        assert torch.equal(
            torch.count_nonzero(prev_output_tokens != pad_idx, dim=1), target_lengths
        )
        assert torch.equal(
            torch.count_nonzero(target_tokens != pad_idx, dim=1), target_lengths
        )
        return prev_output_tokens, target_tokens, target_lengths

    def _get_text_to_units_batch(self, raw_batch: Any) -> SeqsBatch:
        root = raw_batch[self.ROOT_COLUMN]
        seqs = root[self.TARGET_UNITS_COLUMN]
        prefix_tokens = root[self.TARGET_LANG_COLUMN][:, 1, :]
        pad_idx = self.unit_tokenizer.vocab_info.pad_idx
        eos_idx = self.unit_tokenizer.vocab_info.eos_idx
        assert pad_idx is not None
        assert eos_idx is not None

        (
            prev_output_tokens,
            target_tokens,
            target_lengths,
        ) = self._gen_prev_toks_target_toks_target_lens(
            seqs=seqs,
            prefix_tokens=prefix_tokens,
            pad_idx=pad_idx,
            eos_idx=eos_idx,
        )

        return SeqsBatch(
            src_tokens=None,
            src_lengths=None,
            target_tokens=target_tokens.to(self.target_device),
            prev_output_tokens=prev_output_tokens.to(self.target_device),
            target_lengths=target_lengths.to(self.target_device),
            prefix_tokens=prefix_tokens.to(self.target_device),
        )

    def _get_speech_src_tokens_and_lengths(
        self, raw_batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fbanks = raw_batch[self.ROOT_COLUMN][self.AUDIO_COLUMN_NAME]["data"]["fbank"]
        return fbanks["seqs"].to(self.float_dtype), fbanks["seq_lens"]

    def _get_speech_to_text_batch(self, raw_batch: Any) -> SeqsBatch:
        root = raw_batch[self.ROOT_COLUMN]
        seqs = root[self.TARGET_TEXT_COLUMN]
        prefix_tokens = root[self.TARGET_LANG_COLUMN][:, 0, :]
        pad_idx = self.text_tokenizer.vocab_info.pad_idx
        assert pad_idx is not None
        eos_idx = self.text_tokenizer.vocab_info.eos_idx
        assert eos_idx is not None

        (
            prev_output_tokens,
            target_tokens,
            target_lengths,
        ) = self._gen_prev_toks_target_toks_target_lens(
            seqs=seqs,
            prefix_tokens=prefix_tokens,
            pad_idx=pad_idx,
            eos_idx=eos_idx,
        )
        src_tokens, src_lengths = self._get_speech_src_tokens_and_lengths(
            raw_batch=raw_batch
        )

        return SeqsBatch(
            src_tokens=src_tokens.to(self.target_device),
            src_lengths=src_lengths.to(self.target_device),
            target_tokens=target_tokens.to(self.target_device),
            prev_output_tokens=prev_output_tokens.to(self.target_device),
            target_lengths=target_lengths.to(self.target_device),
            prefix_tokens=prefix_tokens.to(self.target_device),
        )

    def _convert_to_mulitmodal_seqs_batch(self, raw_batch: Any) -> MultimodalSeqsBatch:
        return MultimodalSeqsBatch(
            speech_to_text=self._get_speech_to_text_batch(raw_batch=raw_batch),
            text_to_units=self._get_text_to_units_batch(raw_batch=raw_batch),
        )

    def iterate_batches(self) -> Iterator[MultimodalSeqsBatch]:
        for raw_batch in self.pipeline:
            yield self._convert_to_mulitmodal_seqs_batch(raw_batch)

    def reset(self) -> None:
        self.pipeline.reset()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)s -- %(name)s.{os.getpid()}: %(message)s",
    )
    config = DataLoadingConfig(
        audio=AudioProcessingConfig(
            audio_root_dir="/fsx-ust/data/audio_zips/",
        ),
        manifest_path_prefix="/fsx-ust/spopuri/datasets/S2ST/V1/M4T_V1_phase2/primary",
        manifest_list_path="/data/home/mavlyutov/train_manifests.txt",
        shuffle_window=1000,
        num_threads=5,
    )
    loader = UnityDataLoader(config=config, target_device=torch.device("cpu"))
    for idx, batch in enumerate(loader.iterate_batches()):
        if idx % 10 == 0:
            assert batch.speech_to_text.src_tokens is not None
            print(batch.speech_to_text.src_tokens.shape)
            logger.info(f".. pulled {idx} batches")
            if idx > 1000:
                break
