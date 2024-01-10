# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq2.data.audio import AudioDecoderOutput
from fairseq2.nn.padding import get_seqs_and_padding_mask

from seamless_communication.models.conformer_shaw import load_conformer_shaw_model

from tests.common import (
    convert_to_collated_fbank,
    get_default_dtype,
    device,
)

REF_MEAN, REF_STD = -0.0001, 0.1547


def test_conformer_shaw_600m(example_rate16k_audio: AudioDecoderOutput) -> None:

    dtype = get_default_dtype()
    audio_dict = example_rate16k_audio
    src = convert_to_collated_fbank(audio_dict, dtype=dtype)
    seqs, padding_mask = get_seqs_and_padding_mask(src)

    model = load_conformer_shaw_model("conformer_shaw", device=device, dtype=dtype)
    model.eval()

    with torch.inference_mode():
        seqs, padding_mask = model.encoder_frontend(seqs, padding_mask)

        seqs, _ = model.encoder(seqs, padding_mask)

    std, mean = torch.std_mean(seqs)

    assert round(mean.item(), 4) == REF_MEAN
    assert round(std.item(), 4) == REF_STD
