# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import argparse
import logging

from fairseq2.assets import asset_store, download_manager
from seamless_communication.cli.streaming.scorers.seamless_quality_scorer import (
    SeamlessQualityScorer,
)
from seamless_communication.streaming.agents.seamless_s2st import SeamlessS2STAgent
from seamless_communication.streaming.agents.seamless_streaming_s2st import (
    SeamlessStreamingS2STAgent,
)
from seamless_communication.streaming.agents.seamless_streaming_s2t import (
    SeamlessStreamingS2TAgent,
)
from simuleval.cli import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="Streaming evaluation of Seamless UnitY models",
        conflict_handler="resolve",
    )

    parser.add_argument(
        "--task",
        choices=["s2st", "s2tt", "asr"],
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

    args, _ = parser.parse_known_args()

    model_configs = dict(
        source_segment_size=320,
        device="cuda:0",
        dtype="fp16",
        min_starting_wait_w2vbert=192,
        decision_threshold=0.5,
        no_early_stop=True,
        max_len_a=0,
        max_len_b=100,
    )

    eval_configs = dict(quality_metrics="SEAMLESS_QUALITY_SCORER")
    if args.task == "s2st":
        model_configs["min_unit_chunk_size"] = 50
        eval_configs["latency_metrics"] = "StartOffset EndOffset"

        if args.expressive:
            agent_class = SeamlessS2STAgent
        else:
            agent_class = SeamlessStreamingS2STAgent
    elif args.task in ["s2tt", "asr"]:
        assert args.expressive is False, "S2TT inference cannot be expressive."
        agent_class = SeamlessStreamingS2TAgent
        parser.add_argument(
            "--unity-model-name",
            type=str,
            help="Unity model name.",
            default="seamless_streaming_unity",
        )
        args, _ = parser.parse_known_args()
        asset_card = asset_store.retrieve_card(name=args.unity_model_name)
        tokenizer_uri = asset_card.field("tokenizer").as_uri()
        tokenizer_path = download_manager.download_tokenizer(
            tokenizer_uri, asset_card.name, force=False, progress=True
        )
        eval_configs["latency_metrics"] = "AL LAAL"
        eval_configs["eval_latency_unit"] = "spm"
        eval_configs["eval_latency_spm_model"] = tokenizer_path

    base_config = dict(
        dataloader="fairseq2_s2tt",
        dataloader_class="seamless_communication.streaming.dataloaders.s2tt.SimulEvalSpeechToTextDataloader",
    )

    evaluate(agent_class, {**base_config, **model_configs, **eval_configs}, parser)


if __name__ == "__main__":
    main()
