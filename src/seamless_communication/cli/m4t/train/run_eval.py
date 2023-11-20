# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import platform
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Tuple, Union
from zipfile import ZipFile, ZipInfo

import sacrebleu
import torch
import torchaudio  # type: ignore
from jiwer import wer  # type: ignore

import seamless_communication.cli.m4t.train.cleaners as cleaners
from fairseq2.data.audio import WaveformToFbankConverter
from fairseq2.generation import NGramRepeatBlockProcessor, SequenceGeneratorOptions
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from seamless_communication.cli.m4t.train import model as _model
from seamless_communication.cli.m4t.train import trainer as _trainer
from seamless_communication.cli.m4t.train.configs import (
    DataLoadingConfig,
    WorkflowParams,
)
from seamless_communication.inference.generator import UnitYGenerator
from seamless_communication.models.tokenizer import SPMTokenizer
from seamless_communication.models.unity import (
    UnitTokenizer,
    UnitYModel,
    load_unity_text_tokenizer,
    load_unity_unit_tokenizer,
)

logging_format = f"%(asctime)s - {platform.node()} - %(process)s - %(levelname)s - %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=logging_format,
)

logger = logging.getLogger("eval")


class TestRecord(NamedTuple):
    wav: torch.Tensor
    tgt_lang: str
    tgt_text: str


RAW_TGT_TEXT_COL_NAME = "tgt_text"
TGT_LANG_COL_NAME = "tgt_lang"
AUDIO_COL_NAME = "audio"
SAMPLE_RATE = 16000
OPEN_ZIP_ARCHIVES: Dict[str, Tuple[ZipFile, List[ZipInfo]]] = {}


def get_bleu(
    translations: List[str], ref_translations: List[str], dialect: str = "en"
) -> float:
    if dialect.lower().startswith("ja") or dialect.lower().startswith("zh"):
        tokenizer = "char"
    else:
        tokenizer = "13a"
    print(f"Num samples {len(translations)} {len(ref_translations)}")
    for idx in range(5):
        logger.info(f"Example transl: {translations[idx]}")
        logger.info(f"Example refere: {ref_translations[idx]}")
        logger.info("---")
    score = sacrebleu.corpus_bleu(translations, [ref_translations], tokenize=tokenizer)
    return score.score


def _remove_stuttering(text):
    filt = []
    for word in text.split():
        if len(filt) > 1 and filt[-1] == word and filt[-2] == word:
            continue
        filt.append(word)
    return " ".join(filt)


def _normalize_text_for_wer(text, lang="en"):
    text = cleaners.basic_cleaners(text)
    text = cleaners.remove_punctuations(text, cleaners.PUNCTUATIONS_EXCLUDE_APOSTROPHE)
    text = _remove_stuttering(text)
    if lang == "ja":
        text = cleaners.normalize_ja_text(text)
    return text


def get_wer(
    translations: List[str], ref_translations: List[str], dialect: str = "en"
) -> float:
    reference = [_normalize_text_for_wer(txt) for txt in ref_translations]
    hypothesis = [_normalize_text_for_wer(txt) for txt in translations]
    return (
        wer(
            reference=reference,
            hypothesis=hypothesis,
        )
        * 100
    )


def _iter_manifest(manifest_path: Path) -> Iterator[Tuple[str, str, str]]:
    tgt_lang_idx = None
    tgt_text_idx = None
    audio_idx = None
    with open(manifest_path) as fp_in:
        for line in fp_in:
            chunks = line.strip().split("\t")
            if tgt_lang_idx is None:  # header
                tgt_lang_idx = chunks.index(TGT_LANG_COL_NAME)
                tgt_text_idx = chunks.index(RAW_TGT_TEXT_COL_NAME)
                audio_idx = chunks.index(AUDIO_COL_NAME)
                continue
            yield chunks[audio_idx], chunks[tgt_lang_idx], chunks[tgt_text_idx]


def _extract_audio_blob(arch_name: str, offset: int) -> bytes:
    archive, records = OPEN_ZIP_ARCHIVES[arch_name]
    for info in records:
        info_offset = info.header_offset + len(info.FileHeader())
        if abs(info_offset - offset) < 100:  # expect some misalignment
            local_path = archive.extract(info)
            with open(local_path, "rb") as fp_in:
                content_bytes = fp_in.read()
            os.unlink(local_path)
            return content_bytes
    raise ValueError(f"Didn't find record with offset {offset} in {arch_name}")


def _load_archive_data(audio_zips_root: str, name: str) -> None:
    if name in OPEN_ZIP_ARCHIVES:
        return
    archive = ZipFile(
        os.path.join(audio_zips_root, name),
        mode="r",
    )
    OPEN_ZIP_ARCHIVES[name] = (archive, archive.infolist())
    logging.info(f"Loaded archive {name}")


def _load_audio_wav(audio_zips_root: str, audio_str: str) -> torch.Tensor:
    archive_name, offset_str, _ = audio_str.split(":")
    offset = int(offset_str)
    _load_archive_data(audio_zips_root=audio_zips_root, name=archive_name)
    blob = _extract_audio_blob(arch_name=archive_name, offset=offset)
    wav, samplerate = torchaudio.load(BytesIO(blob))
    assert samplerate == SAMPLE_RATE
    return wav


def load_manifest(manifest_path: Path, audio_zips_root: str) -> Iterator[TestRecord]:
    for audio_str, tgt_lang, tgt_text in _iter_manifest(manifest_path=manifest_path):
        audio = _load_audio_wav(audio_zips_root=audio_zips_root, audio_str=audio_str)
        yield TestRecord(wav=audio, tgt_lang=tgt_lang, tgt_text=tgt_text)


def _init_unit_tokenizer(data_config: DataLoadingConfig) -> UnitTokenizer:
    if data_config.unit_tokenization.from_model is not None:
        return load_unity_unit_tokenizer(data_config.unit_tokenization.from_model)
    else:
        raise NotImplementedError("TBD")


def _init_text_tokenizer(
    data_config: DataLoadingConfig,
) -> Union[NllbTokenizer, SPMTokenizer]:
    if data_config.text_tokenization.from_model is not None:
        return load_unity_text_tokenizer(data_config.text_tokenization.from_model)
    else:
        assert data_config.text_tokenization.langtoks is not None
        assert data_config.text_tokenization.spm_path is not None
        return SPMTokenizer(
            pathname=data_config.text_tokenization.spm_path,
            langs=data_config.text_tokenization.langtoks,
        )


def translate(
    model: UnitYModel,
    text_tokenizer: Union[NllbTokenizer, SPMTokenizer],
    unit_tokenizer: UnitTokenizer,
    fbank_extractor: WaveformToFbankConverter,
    dtype: torch.dtype,
    device: torch.device,
    test_record: TestRecord,
    ngram_filtering: bool = True,
    text_max_len_a: int = 1,
    text_max_len_b: int = 200,
    unit_max_len_a: int = 1,
    unit_max_len_b: int = 50,
) -> Tuple[str, Any]:
    """Runs S2T translation. TBD: add S2S"""
    text_opts = SequenceGeneratorOptions(
        beam_size=5, soft_max_seq_len=(text_max_len_a, text_max_len_b)
    )
    unit_opts = SequenceGeneratorOptions(
        beam_size=5, soft_max_seq_len=(unit_max_len_a, unit_max_len_b)
    )
    if ngram_filtering:
        text_opts.step_processor = NGramRepeatBlockProcessor(ngram_size=4)
        unit_opts.step_processor = NGramRepeatBlockProcessor(ngram_size=4)
    generator = UnitYGenerator(
        model,
        text_tokenizer,
        test_record.tgt_lang,
        unit_tokenizer,
        text_opts=text_opts,
        unit_opts=unit_opts,
    )
    wav = test_record.wav
    assert len(wav.shape) in (1, 2)
    if len(wav.shape) == 1:
        wav = wav.unsqueeze(-1)
    elif wav.shape[0] <= 2:  # channel is first, should be second:
        wav = wav.transpose(0, 1)
    fbank = fbank_extractor(
        {
            "waveform": wav,
            "sample_rate": SAMPLE_RATE,
        }
    )["fbank"]
    s2t_result, t2u_result = generator(
        fbank.unsqueeze(0),
        None,
        "speech",
        "text",
        ngram_filtering=ngram_filtering,
    )
    s2t_out = str(s2t_result.sentences[0])
    return s2t_out, None


def run_evaluation(
    parameters: WorkflowParams, checkpoint_path: Path, manifest_path: Path
):
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    float_dtype = _trainer.UnitYTrainer._get_float_dtype(
        parameters.training.float_dtype
    )
    logger.info(f"Device: {device}, float dtype: {float_dtype}")
    audio_zips_root = parameters.train_data.audio.audio_root_dir
    logger.info(f"Audio zip root: {audio_zips_root}")
    model = _model.ModelBuilder(
        config=parameters.model, dtype=float_dtype, device=device
    ).build_model(skip_loading_weights=True)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    # temporary fix for previous bug with checkpoint saving:
    state_dict = {
        _trainer.UnitYTrainer._strip_state_key_prefixes(
            key.replace("t2u.", "t2u_model.")
        ): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    text_tokenizer = _init_text_tokenizer(data_config=parameters.train_data)
    unit_tokenizer = _init_unit_tokenizer(data_config=parameters.train_data)
    fbank_extractor = WaveformToFbankConverter(
        num_mel_bins=parameters.train_data.audio.fbanks_num_mel_bins or 80,
        waveform_scale=parameters.train_data.audio.fbanks_waveform_scale,
        channel_last=True,
        standardize=parameters.train_data.audio.fbanks_standardize_audio,
        device=device,
        dtype=float_dtype,
    )

    logger.info(f"Model: {model}")
    records = load_manifest(
        manifest_path=manifest_path, audio_zips_root=audio_zips_root
    )

    model_translations = []
    reference_translations = []
    for idx, record in enumerate(records):
        reference_translations.append(record.tgt_text)
        s2t, t2u = translate(
            model=model,
            text_tokenizer=text_tokenizer,
            unit_tokenizer=unit_tokenizer,
            fbank_extractor=fbank_extractor,
            test_record=record,
            device=device,
            dtype=float_dtype,
        )
        model_translations.append(s2t)
        logger.info(f"{idx} ref: {record.tgt_text}")
        logger.info(f"{idx} s2t: {s2t}")
        logger.info("--")
    model_wer = get_wer(model_translations, ref_translations=reference_translations)
    logger.info(f"FINAL WER: {model_wer}, manifest: {manifest_path}")


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M4T training")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to test manifest",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--train_params",
        type=Path,
        required=True,
        help="Training workflow config (*_config.yaml is available in work directory)",
    )
    return parser


def main() -> None:
    args = init_parser().parse_args()
    manifest: Path = args.manifest
    config_path: Path = args.train_params
    checkpoint_path: Path = args.checkpoint
    assert manifest.exists()
    assert config_path.exists()
    assert checkpoint_path.exists()
    parameters = WorkflowParams.from_file(config_path.as_posix())
    run_evaluation(
        parameters=parameters, checkpoint_path=checkpoint_path, manifest_path=manifest
    )


if __name__ == "__main__":
    main()
