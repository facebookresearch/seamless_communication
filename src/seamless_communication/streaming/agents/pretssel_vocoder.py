# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
from __future__ import annotations

import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List

import torch
from fairseq2.assets import asset_store
from fairseq2.data.audio import WaveformToFbankConverter, WaveformToFbankInput
from seamless_communication.models.generator.loader import load_pretssel_vocoder_model
from seamless_communication.models.unity import load_gcmvn_stats
from seamless_communication.store import add_gated_assets
from seamless_communication.streaming.agents.common import (
    AgentStates,
    NoUpdateTargetMixin,
)
from simuleval.agents import TextToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s -- %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class PretsselVocoderAgent(NoUpdateTargetMixin, TextToSpeechAgent):  # type: ignore
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

        if args.gated_model_dir:
            add_gated_assets(args.gated_model_dir)

        logger.info(
            f"Loading the Vocoder model: {args.vocoder_name} on device={args.device}, dtype={args.dtype}"
        )
        assert "pretssel" in args.vocoder_name
        self.vocoder = load_pretssel_vocoder_model(
            args.vocoder_name, device=args.device, dtype=args.dtype
        )
        self.vocoder.eval()

        vocoder_model_card = asset_store.retrieve_card(args.vocoder_name)
        self.vocoder_sample_rate = vocoder_model_card.field("sample_rate").as_(int)

        self.upstream_idx = args.upstream_idx
        self.sample_rate = args.sample_rate  # input sample rate
        self.tgt_lang = args.tgt_lang
        self.convert_to_fbank = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=False,
            device=args.device,
            dtype=args.dtype,
        )

        _gcmvn_mean, _gcmvn_std = load_gcmvn_stats(args.vocoder_name)
        self.gcmvn_mean = torch.tensor(
            _gcmvn_mean, device=args.device, dtype=args.dtype
        )
        self.gcmvn_std = torch.tensor(_gcmvn_std, device=args.device, dtype=args.dtype)

    def gcmvn_normalize(self, seqs: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = seqs.subtract(self.gcmvn_mean).divide(self.gcmvn_std)
        return result

    @torch.inference_mode()
    def policy(self, states: AgentStates) -> WriteAction:
        """
        The policy is always write if there is a waveform
        """
        units = states.source

        if len(units) == 0 or len(units[0]) == 0:
            if states.source_finished:
                return WriteAction(content=[], finished=True)
            else:
                return ReadAction()

        unit = units[0][0]

        # adjust the control symbols for the embedding
        unit += 4

        unit, duration = torch.unique_consecutive(unit, return_counts=True)

        duration *= 2

        if isinstance(states.upstream_states[self.upstream_idx].source, list):
            source: List[float] = sum(
                states.upstream_states[self.upstream_idx].source, []
            )
        else:
            source = states.upstream_states[self.upstream_idx].source

        audio_dict: WaveformToFbankInput = {
            "waveform": torch.tensor(
                source, dtype=torch.float32, device=self.device
            ).unsqueeze(1),
            "sample_rate": self.sample_rate,
        }

        feats = self.convert_to_fbank(audio_dict)["fbank"]

        feats = self.gcmvn_normalize(feats)

        tgt_lang = states.tgt_lang if states.tgt_lang else self.tgt_lang

        wav = self.vocoder(
            unit,
            tgt_lang=tgt_lang,
            prosody_input_seqs=feats,
            durations=duration.unsqueeze(0),
            normalize_before=True,
        )

        states.source = []

        return WriteAction(
            SpeechSegment(
                content=wav[0][0][0].tolist(),
                finished=states.source_finished,
                sample_rate=self.vocoder_sample_rate,
                tgt_lang=tgt_lang,
            ),
            finished=states.source_finished,
        )

    @classmethod
    def add_args(cls, parser: ArgumentParser) -> None:
        param = parser.add_argument(
            "--gated-model-dir",
            type=Path,
            required=False,
            help="SeamlessExpressive model directory.",
        )
        parser.add_argument(
            "--vocoder-name",
            type=str,
            help="Vocoder name - vocoder_pretssel or vocoder_pretssel_16khz",
            default="vocoder_pretssel",
        )
        parser.add_argument(
            "--upstream-idx",
            type=int,
            default=0,
            help="index of encoder states where states.source contains input audio",
        )

    @classmethod
    def from_args(
        cls, args: Namespace, **kwargs: Dict[str, Any]
    ) -> PretsselVocoderAgent:
        return cls(args)
