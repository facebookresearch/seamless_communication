# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from urllib.request import urlretrieve

import torch
import torchaudio

from seamless_communication.models.vocoder.loader import load_mel_vocoder_model
from tests.common import assert_close, device


def test_pretssel_vocoder() -> None:
    n_mel_bins = 80
    sample_rate = 16_000

    vocoder = load_mel_vocoder_model(
        "vocoder_mel", device=device, dtype=torch.float32
    )

    url = "https://keithito.com/LJ-Speech-Dataset/LJ037-0171.wav"

    with tempfile.NamedTemporaryFile() as f:
        urlretrieve(url, f.name)
        _wav, _sr = torchaudio.load(f.name)

    wav = torchaudio.sox_effects.apply_effects_tensor(
        _wav, _sr, [["rate", f"{sample_rate}"], ["channels", "1"]]
    )[0].to(device=device)
    feat = torchaudio.compliance.kaldi.fbank(
        wav * (2**15), num_mel_bins=n_mel_bins, sample_frequency=sample_rate
    )

    with torch.no_grad():
        wav_hat = vocoder(feat).t()

    feat_hat = torchaudio.compliance.kaldi.fbank(
        wav_hat * (2**15), num_mel_bins=n_mel_bins, sample_frequency=sample_rate
    )

    assert_close(feat_hat, feat[: feat_hat.shape[0], :], atol=0.0, rtol=5.0)


if __name__ == "__main__":
    test_pretssel_vocoder()
