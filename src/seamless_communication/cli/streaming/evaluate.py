# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from fairseq2.assets import asset_store, download_manager
from seamless_communication.cli.eval_utils import get_tokenizer
from seamless_communication.cli.streaming.scorers.seamless_whisper_asr_bleu import (
    SeamlessWhisperASRSacreBLEUScorer as SeamlessWhisperASRSacreBLEUScorer,
)
from seamless_communication.streaming.agents.mma_m4t_s2st import (
    MonotonicM4TS2STAgent,
    SeamlessS2STAgent,
)
from seamless_communication.streaming.agents.mma_m4t_s2t import MonotonicM4TS2TAgent
from simuleval.evaluator import build_evaluator
from simuleval.utils.agent import EVALUATION_SYSTEM_LIST, build_system_args


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Streaming evaluation of Seamless UnitY models",
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--task",
        choices=["s2st", "s2tt"],
        required=True,
        type=str,
        help="Target language to translate/transcribe into.",
    )
    parser.add_argument(
        "--expressive",
        action="store_true",
        default=False,
        help="Expressive streaming S2ST inference",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
    )

    args, _ = parser.parse_known_args()

    model_configs = dict(
        source_segment_size=320,
        device="cuda:0",
        dtype=args.dtype,
        min_starting_wait_w2vbert=192,
        decision_threshold=0.5,
        no_early_stop=True,
        max_len_a=1,
        max_len_b=200,
    )

    if args.dtype == "fp16":
        model_configs.update(dict(fp16=True))

    EVALUATION_SYSTEM_LIST.clear()
    if args.task == "s2st":
        model_configs.update(
            dict(
                min_unit_chunk_size=50,
            )
        )
        eval_configs = dict(
            quality_metrics="SEAMLESS_WHISPER_ASR_BLEU",
            latency_metrics="StartOffset EndOffset",
            whisper_model_size="large-v2",
            normalize_asr_bleu_references=True,
        )
        if args.expressive:
            EVALUATION_SYSTEM_LIST.append(SeamlessS2STAgent)
            model_configs.update(dict(vocoder_name="vocoder_pretssel"))
        else:
            EVALUATION_SYSTEM_LIST.append(MonotonicM4TS2STAgent)
    elif args.task == "s2tt":
        EVALUATION_SYSTEM_LIST.append(MonotonicM4TS2TAgent)
        parser.add_argument(
            "--unity-model-name",
            type=str,
            help="Unity model name.",
            default="seamless_streaming_unity",
        )
        parser.add_argument(
            "--tgt-lang",
            default="eng",
            type=str,
            help="Target language to translate/transcribe into.",
        )
        args, _ = parser.parse_known_args()
        asset_card = asset_store.retrieve_card(name=args.unity_model_name)
        tokenizer_uri = asset_card.field("tokenizer").as_uri()
        tokenizer_path = download_manager.download_tokenizer(
            tokenizer_uri, asset_card.name, force=False, progress=True
        )
        eval_configs = dict(
            sacrebleu_tokenizer=get_tokenizer(args.tgt_lang),
            eval_latency_unit="spm",
            eval_latency_spm_model=tokenizer_path,
            latency_metrics="AL LAAL",
        )

    base_config = dict(
        dataloader="fairseq2_s2tt",
        dataloader_class="seamless_communication.streaming.dataloaders.s2tt.SimulEvalSpeechToTextDataloader",
    )

    system, args = build_system_args(
        {**base_config, **model_configs, **eval_configs}, parser
    )

    evaluator = build_evaluator(args)
    evaluator(system)


if __name__ == "__main__":
    main()
