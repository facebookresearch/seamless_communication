# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from fairseq2.data.audio import AudioDecoderOutput, WaveformToFbankInput

from seamless_communication.models.vocoder.loader import load_mel_vocoder_model
from tests.common import (
    assert_close,
    convert_to_collated_fbank,
    device,
    get_default_dtype,
)


def test_pretssel_vocoder(example_rate16k_audio: AudioDecoderOutput) -> None:
    sample_rate = 16_000

    dtype = get_default_dtype()

    audio_dict = example_rate16k_audio

    feat = convert_to_collated_fbank(audio_dict, dtype=dtype)["seqs"][0]

    vocoder = load_mel_vocoder_model("vocoder_mel", device=device, dtype=dtype)
    vocoder.eval()

    with torch.inference_mode():
        wav_hat = vocoder(feat).view(1, -1)

    audio_hat = {"sample_rate": sample_rate, "waveform": wav_hat}

    audio_hat_dict = cast(WaveformToFbankInput, audio_hat)

    feat_hat = convert_to_collated_fbank(audio_hat_dict, dtype=dtype)["seqs"][0]

    assert_close(feat_hat, feat[: feat_hat.shape[0], :], atol=0.0, rtol=5.0)
