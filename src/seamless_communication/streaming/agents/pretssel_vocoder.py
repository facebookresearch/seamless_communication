# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List

import torch
from fairseq2.data.audio import WaveformToFbankConverter, WaveformToFbankInput
from seamless_communication.models.generator.vocoder import PretsselVocoder
from seamless_communication.models.unity import load_gcmvn_stats
from seamless_communication.models.vocoder.vocoder import Vocoder
from seamless_communication.streaming.agents.common import NoUpdateTargetMixin
from simuleval.agents import AgentStates, TextToSpeechAgent
from simuleval.agents.actions import ReadAction, WriteAction
from simuleval.data.segments import SpeechSegment


class PretsselVocoderAgent(NoUpdateTargetMixin, TextToSpeechAgent):  # type: ignore
    def __init__(self, vocoder: Vocoder, args: Namespace) -> None:
        super().__init__(args)
        self.vocoder = vocoder
        self.upstream_idx = args.upstream_idx
        self.sample_rate = args.sample_rate  # input sample rate
        self.vocoder_sample_rate = args.vocoder_sample_rate  # output sample rate
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
        parser.add_argument(
            "--upstream-idx",
            type=int,
            default=0,
            help="index of encoder states where states.source contains input audio",
        )
        parser.add_argument(
            "--vocoder-sample-rate",
            type=int,
            default=16000,
            help="sample rate out of the vocoder",
        )

    @classmethod
    def from_args(
        cls, args: Namespace, **kwargs: Dict[str, Any]
    ) -> PretsselVocoderAgent:
        vocoder = kwargs.get("vocoder", None)
        assert isinstance(vocoder, PretsselVocoder)
        return cls(vocoder, args)
