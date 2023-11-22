# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from seamless_communication.cli.streaming.scorers.seamless_whisper_asr_bleu import (
    SeamlessWhisperASRSacreBLEUScorer as SeamlessWhisperASRSacreBLEUScorer,
)
from seamless_communication.streaming.agents import MonotonicM4TS2STAgent
from simuleval.cli import evaluate


if __name__ == "__main__":
    tgt_lang = "eng"

    data_configs = dict(
        dataloader="fairseq2_s2tt",
        dataloader_class="seamless_communication.streaming.dataloaders.s2tt.SimulEvalSpeechToTextDataloader",
        data_file="/large_experiments/seamless/ust/annaysun/datasets/s2ut_pt/x2t_v2/dev_fleurs_spa-eng.tsv",
        tgt_lang=tgt_lang,
        audio_root_dir="/large_experiments/seamless/ust/data/audio_zips",
        end_index=10,
    )

    model_configs = dict(
        agent_class="seamless_communication.streaming.agents.mma_m4t_s2st.MonotonicM4TS2STAgent",
        source_segment_size=320,
        task="s2st",
        device="cuda:0",
        dtype="fp16",
        min_starting_wait_w2vbert=192,
        decision_threshold=0.5,
        min_unit_chunk_size=50,
        no_early_stop=True,
        max_len_a=0,
        max_len_b=100,
    )

    eval_configs = dict(
        output=f"MonotonicM4TS2STAgent_spa-eng_debug",
        quality_metrics="SEAMLESS_WHISPER_ASR_BLEU",
        latency_metrics="StartOffset EndOffset",
        whisper_model_size="large-v2",
        normalize_asr_bleu_references=True,
    )

    evaluate(MonotonicM4TS2STAgent, {**data_configs, **model_configs, **eval_configs})
