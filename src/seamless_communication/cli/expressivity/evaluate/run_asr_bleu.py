# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Tuple, Union
import logging
import pandas as pd
import whisper
from whisper import Whisper
from sacrebleu.metrics.base import Score, Signature
from fire import Fire
from seamless_communication.cli.eval_utils.compute_metrics import (
    init_whisper_model,
    compute_corpus_metric_score,
    transcribe_series,
)
from fairseq2.typing import Device


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


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
    if lang == "cmn":
        try:
            import chinese_converter
        except ImportError:
            raise ImportError(
                "Please install chinese_converter: pip install chinese_converter"
            )
    audio_transcriptions = transcribe_series(
        audio_paths_series,
        asr_model,
        audio_lang=lang,
        beam_size=beam_size,
        temperature=temperature,
    )
    if lang == "cmn":
        audio_transcriptions = pd.Series([chinese_converter.to_simplified(text) for text in audio_transcriptions])
        ref_text_series = pd.Series([chinese_converter.to_simplified(text) for text in ref_text_series])
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


def compute_s2st_quality_metrics(
    output_manifest_tsv_path: Path,
    output_path: Path,
    tgt_lang: str,
    device: Device,
    whisper_model_name: str = "large",
    whisper_normalize_text_output: bool = False,
    ref_text_col_name: str = "ref_tgt_text",
    pred_text_col_name: Optional[str] = "pred_tgt_text",
    pred_audio_col_name: str = "pred_tgt_audio",
) -> str:
 
    df = pd.read_csv(
        output_manifest_tsv_path, sep="\t", quoting=3, encoding="utf-8", escapechar="\\"
    )

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    metric = "bleu"

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

    filename = (
        "s2tt_bleu_normalized.json"
        if whisper_normalize_text_output
        else "s2tt_bleu.json"
    )
    cur_task = "S2TT"

    with open(output_path / filename, "w") as f:
        f.write(text_metric_json)

    logger.info(f"{cur_task} {metric}:\n{text_metric_json}")

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
    filename = "s2st_asr_bleu_normalized.json"

    with open(
        output_path / filename,
        "w",
    ) as f:
        f.write(asr_bleu_normalized_json)

    logger.info(f"S2ST ASR Normalized BLEU:\n{asr_bleu_normalized_json}")

    return filename


def run_asr_bleu_expressive_model(
    generation_dir_path: str,
    generate_tsv_filename: str,
    tgt_lang: str,
):
    compute_s2st_quality_metrics(
        f"{generation_dir_path}/{generate_tsv_filename}",
        Path(generation_dir_path),
        tgt_lang,
        device=Device("cuda"),
        ref_text_col_name="tgt_text",
        pred_text_col_name="s2t_out",
        pred_audio_col_name="hypo_audio",
    )


if __name__ == "__main__":
    Fire(run_asr_bleu_expressive_model)
