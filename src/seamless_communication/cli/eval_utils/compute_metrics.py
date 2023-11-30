# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import whisper
from fairseq2.typing import Device
from jiwer import cer, wer
from sacrebleu.metrics.base import Score, Signature
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from seamless_communication.cli.eval_utils.lang_mapping import LANG3_LANG2
from tqdm import tqdm
from whisper import Whisper
from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def init_whisper_model(
    device: Device,
    whisper_model_name: str = "large",
) -> Whisper:
    return whisper.load_model(name=whisper_model_name, device=device)


def transcribe_series(
    audio_paths_series: pd.Series,
    asr_model: Whisper,
    audio_lang: str,
    beam_size: int = 1,
    temperature: float = 0.0,
) -> pd.Series:
    """Transcribes each audio filepath from series and returns series of transcriptions
    Args:
        audio_paths_series (pd.Series): each line contains path to audio file.
        asr_model: ASR model to do the transcribing process e.g. Whisper
        audio_lang (str): what language is used in the given audio, used by ASR model
        beam_size (int): whisper beam size. Defaults to 1
        temperature (float): whisper temperature. Defaults to 0.0 to avoid fallback decoding (see details below).
    Returns:
        pd.Series: Series where each line has a transcription of corresponding audio from audio_paths_series
    Whisper model implements decoding with fallback: https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L147
    The core idea is that decoding at each time step might happen multiple times if at least one criterion to "fall back" i.e.
    start over is fired. Number of fallback iterations is determined by the schedule of temperature values:
    https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L41
    By default this schedule is active and temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0) i.e. even with beam_size 5 it might fell back and
    turn on sampling by using temperature > 0, in this case the beam search is not used in the fall back iteration.
    Explicit setting of temperature=0.0 overwrites the schedule and fall back decoding has only one for loop iteration i.e. no fall backs.
    This allows us to do reproducible evaluation without sample variations. Beware that this might introduce the repetition loops in
    the transcriptions and lead to worse ASR-BLEU score in the end.
    """

    if len(audio_lang) == 3:
        # to make it work with whisper
        audio_lang = LANG3_LANG2[audio_lang]

    transcriptions = {}

    for idx, audio_path in tqdm(
        audio_paths_series.items(),
        desc=f"Transcribing {audio_paths_series.name} column",
        total=len(audio_paths_series),
    ):
        hypo = asr_model.transcribe(
            audio_path,
            temperature=temperature,
            beam_size=beam_size,
            language=audio_lang,
        )["text"].strip()
        transcriptions[idx] = hypo

    transcriptions_series = pd.Series(transcriptions)
    transcriptions_series.name = f"{audio_paths_series.name}_transcribed"

    return transcriptions_series


def whisper_normalize_series(
    transcription_series: pd.Series, text_lang: str
) -> pd.Series:
    """Normalizes the text series using whisper noramlizer. English has a specific one in whisper package.
    Args:
        transcription_series (pd.Series): Each line contains arbitrary text written in text_lang
        text_lang (str): Language of the text in series
    Returns:
        pd.Series: Series with normalized text
    """
    if text_lang == "eng":
        normalizer = EnglishTextNormalizer()
    else:
        normalizer = BasicTextNormalizer()

    norm_transcriptions = {}

    for idx, text in transcription_series.items():
        norm_transcriptions[idx] = normalizer(text)

    norm_transcriptions_series = pd.Series(norm_transcriptions)
    norm_transcriptions_series.name = transcription_series.name

    return norm_transcriptions_series


def compute_asr_bleu(
    audio_paths_series: pd.Series,
    ref_text_series: pd.Series,
    lang: str,
    asr_model: Whisper,
    whisper_normalize_text: bool = True,
    beam_size: int = 1,
    temperature: float = 0.0,
    return_transcriptions: bool = True,
) -> Tuple[Score, Signature, pd.DataFrame]:
    """Wraps functions above to compute corpus-level ASR-BLEU
    ASR decoding hyper-parameters are hard coded to ensure reproducibility across evaluations
    Args:
        audio_paths_series (pd.Series): each line contains path to audio
        ref_text_series (pd.Series): each line contains the text reference to compare audio with
        lang (str): the language of both audio and ref_text
        asr_model: whisper ASR model
        whisper_normalize_text (bool): normalize both text hypotheses and reference if True. Defaults to True.
        beam_size (int): beam_size for whisper generation
        temperature (float): Temperature sampling value for whisper generation
        return_transcriptions (bool)
    """

    audio_transcriptions = transcribe_series(
        audio_paths_series,
        asr_model,
        audio_lang=lang,
        beam_size=beam_size,
        temperature=temperature,
    )
    asr_bleu, asr_bleu_signature = compute_corpus_metric_score(
        audio_transcriptions, ref_text_series, lang, whisper_normalize_text
    )
    asr_bleu_signature.info["whisper_asr_beam_size"] = beam_size
    asr_bleu_signature.info["whisper_asr_temperature"] = temperature
    asr_bleu_signature.info["whisper_asr_language"] = lang

    transcript_df = None
    if return_transcriptions:
        transcript_df = pd.concat(
            [
                audio_paths_series,
                audio_transcriptions,
                ref_text_series,
            ],
            axis=1,
            keys=["audio", "transcript", "reference"],
        )
    return asr_bleu, asr_bleu_signature, transcript_df


def get_tokenizer(lang: str, metric: str = "bleu") -> str:
    """Get tokenizer for language
    Args:
        lang (str): Three letter code of the language
        metric (str): Metric being computed. Valid values are "bleu" and "asr"
    """
    lang_tok_map = {
        "cmn": "char",
        "jpn": "char",
        "tha": "char",
        "lao": "char",
        "mya": "char",
    }
    default = (
        "13a" if metric == "bleu" else "word"
    )  # 13a is the default tokenizer for bleu and wer for asr
    tok = lang_tok_map.get(lang, default)
    return tok


def compute_asr_error_rate(
    hyp_text_series: pd.Series,
    ref_text_series: pd.Series,
    lang: str,
    whisper_normalize_text: bool = True,
) -> Tuple[float, str]:
    """Wraps normalization functions and computes ASR WER/CER score
    Args:
        hyp_text_series (pd.Series): each line contains s2t model prediction or first pass prediction
        ref_text_series (pd.Series): _description_
        lang (str): _description_
        whisper_normalize_text (bool, optional): normalize both text hypotheses and reference if True. Defaults to True.
    Returns:
        (MetricScore, MetricScoreSignature)
    """
    if whisper_normalize_text:
        hyp_text_series = whisper_normalize_series(hyp_text_series, lang)
        ref_text_series = whisper_normalize_series(ref_text_series, lang)

    tokenizer_name = get_tokenizer(lang, metric="error_rate")
    metric_name = wer if tokenizer_name == "word" else cer
    metric_score = metric_name(hyp_text_series.to_list(), ref_text_series.to_list())
    return metric_score, f"{metric_name.__name__} is {metric_score}"


def compute_corpus_metric_score(
    hyp_text_series: pd.Series,
    ref_text_series: pd.Series,
    lang: str,
    whisper_normalize_text: bool = True,
    metric: str = "bleu",
) -> Tuple[Score, Signature]:
    """Wraps normalization functions and compute corpus-level BLEU/chrF++ score
    Args:
        hyp_text_series (pd.Series): each line contains s2t model prediction or first pass prediction
        ref_text_series (pd.Series): _description_
        lang (str): _description_
        whisper_normalize_text (bool, optional): normalize both text hypotheses and reference if True. Defaults to True.
    Returns:
        (MetricScore, MetricScoreSignature)
    """
    if whisper_normalize_text:
        hyp_text_series = whisper_normalize_series(hyp_text_series, lang)
        ref_text_series = whisper_normalize_series(ref_text_series, lang)

    tokenizer_name = get_tokenizer(lang)
    corpus_metric_score_metric: Union[BLEU, CHRF]
    if metric == "bleu":
        corpus_metric_score_metric = BLEU(
            lowercase=whisper_normalize_text, tokenize=tokenizer_name
        )  # lowercase applied if we use whisper_normalize_text
    elif metric == "chrF++":
        corpus_metric_score_metric = CHRF(word_order=2)

    corpus_metric_score = corpus_metric_score_metric.corpus_score(
        hyp_text_series.to_list(), [ref_text_series.to_list()]
    )
    corpus_metric_score_signature = corpus_metric_score_metric.get_signature()
    corpus_metric_score_signature.info["whisper_normalize"] = whisper_normalize_text

    return corpus_metric_score, corpus_metric_score_signature


def compute_quality_metrics(
    output_manifest_tsv_path: Path,
    output_path: Path,
    tgt_lang: str,
    task: str,
    device: Device,
    whisper_model_name: str = "large",
    whisper_normalize_text_output: bool = False,
    ref_text_col_name: str = "ref_tgt_text",
    pred_text_col_name: Optional[str] = "pred_tgt_text",
    pred_audio_col_name: str = "pred_tgt_audio",
) -> str:
    """Wraps asr and s2t bleu functions to call it with TSV manifest composed on expressivity side
    Args:
        output_manifest_tsv_path (Path): output manifest which has "ref_text", "hypo_audio", "s2t_out" column names
        output_path (Path): Directory to write files with metrics
        tgt_lang (str): what language we evaluate on
        task (str): Task we are currently evaluating for
        device (Device): Device to use for inference
        whisper_model_name (str): Whisper model name. Defaults to "large".
        whisper_normalize_text_output (bool): Normalizes text output using whisper_normalizer if set to true
        ref_text_col_name (str): Column name in the tsv corresponding to reference target text
        pred_text_col_name (str): Column name in the tsv corresponding to predicted target text
        pred_audio_col_name (str): Column name in the tsv corresponding to predicted target audio.
            Setting this value to none will skip speech metrics
    """
    df = pd.read_csv(
        output_manifest_tsv_path, sep="\t", quoting=3, encoding="utf-8", escapechar="\\"
    )
    task = task.upper()

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    if task in ["S2TT", "S2ST", "T2TT"] and pred_text_col_name:
        metric = "chrF++" if task == "T2TT" else "bleu"
        text_metric, text_metric_signature = compute_corpus_metric_score(
            hyp_text_series=df[pred_text_col_name],
            ref_text_series=df[ref_text_col_name],
            lang=tgt_lang,
            whisper_normalize_text=whisper_normalize_text_output,
            metric=metric,
        )
        text_metric_json = text_metric.format(
            signature=text_metric_signature.format(), is_json=True
        )

        if task == "T2TT":
            filename = "t2tt_chrf.json"
            cur_task = "T2TT"
        else:
            filename = (
                "s2tt_bleu_normalized.json"
                if whisper_normalize_text_output
                else "s2tt_bleu.json"
            )
            cur_task = "S2TT"

        with open(output_path / filename, "w") as f:
            f.write(text_metric_json)

        logger.info(f"{cur_task} {metric}:\n{text_metric_json}")

    if task in ["T2ST", "S2ST"]:
        whisper_model = init_whisper_model(device, whisper_model_name)
        (
            asr_bleu_normalized,
            asr_bleu_normalized_signature,
            transcripts_df,
        ) = compute_asr_bleu(
            audio_paths_series=df[pred_audio_col_name],
            ref_text_series=df[ref_text_col_name],
            lang=tgt_lang,
            asr_model=whisper_model,
            whisper_normalize_text=True,
        )
        transcripts_df.to_csv(
            (output_path / "whisper_audio_transcriptions.tsv"),
            sep="\t",
            index=False,
            encoding="utf-8",
            escapechar="\\",
        )

        asr_bleu_normalized_signature.info["whisper_asr_model"] = whisper_model_name

        asr_bleu_normalized_json = asr_bleu_normalized.format(
            signature=asr_bleu_normalized_signature.format(), is_json=True
        )
        filename = f"{task.lower()}_asr_bleu_normalized.json"

        with open(
            output_path / filename,
            "w",
        ) as f:
            f.write(asr_bleu_normalized_json)

        logger.info(f"{task} ASR Normalized BLEU:\n{asr_bleu_normalized_json}")

    if task == "ASR":
        asr_error_rate, asr_error_rate_signature = compute_asr_error_rate(
            hyp_text_series=df[pred_text_col_name],
            ref_text_series=df[ref_text_col_name],
            lang=tgt_lang,
            whisper_normalize_text=whisper_normalize_text_output,
        )
        d = {
            "name": "WER",
            "score": asr_error_rate,
            "signature": asr_error_rate_signature,
        }
        asr_error_rate_json = json.dumps(d, indent=1, ensure_ascii=False)

        filename = "asr_error_rate.json"

        with open(output_path / filename, "w") as f:
            f.write(asr_error_rate_json)

        logger.info(f"ASR : {asr_error_rate_json}")

    return filename
