# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Final

import torch
from torch import tensor

from fairseq2.data.audio import AudioDecoderOutput
from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor
from tests.common import assert_equal, device, get_default_dtype


REF_TEXT = "the examination and testimony of the experts enabled the commision to conclude that five shots may have been fired"

# fmt: off
REF_DURATIONS_FP16: Final = [[ 1,  1,  2,  1,  1,  5,  5,  6,  4,  3,  2,  3,  4,  4,  2,  2,  2,  1,
          1,  1,  3,  3,  3,  4,  3,  3,  3,  4,  4,  3,  2,  2,  1,  1,  1,  1,
          2,  4,  6,  5,  4,  3,  4,  5,  5, 16,  6,  3,  5,  5,  3,  3,  1,  2,
          1,  1,  1,  2,  3,  2,  3,  1,  3,  3,  3,  2,  2,  4,  2,  2,  2,  3,
          2,  4,  5,  4,  5,  8,  3, 17,  2,  2,  3,  2,  5,  4,  6,  3,  1,  1,
          4,  4,  3,  5,  3,  3,  2,  2,  2,  2,  2,  2,  2,  1,  2,  2,  1,  1,
          2,  6,  4,  5,  9,  5,  1, 12]]
# fmt: on

# fmt: off
REF_DURATIONS_FP32: Final = [[ 1,  1,  2,  1,  1,  5,  5,  6,  4,  3,  2,  3,  4,  4,  2,  2,  2,  1,
           1,  1,  3,  3,  3,  4,  3,  3,  4,  3,  4,  3,  2,  2,  1,  1,  1,  1,
           2,  4,  6,  5,  4,  3,  4,  5,  5, 16,  6,  3,  5,  5,  3,  3,  1,  2,
           1,  1,  1,  2,  3,  2,  3,  1,  3,  3,  3,  2,  2,  4,  2,  2,  2,  3,
           2,  4,  5,  4,  5,  8,  3, 17,  2,  2,  3,  2,  5,  4,  6,  3,  1,  1,
           4,  4,  3,  5,  3,  3,  2,  2,  2,  2,  2,  2,  2,  1,  2,  2,  1,  1,
           2,  6,  4,  5,  9,  5,  1, 12]]
# fmt: on


def test_aligner(example_rate16k_audio: AudioDecoderOutput) -> None:
    aligner_name = "nar_t2u_aligner"
    unit_extractor_name = "xlsr2_1b_v2"
    unit_extractor_output_layer_n = 35
    unit_extractor_kmeans_uri = "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy"
    dtype = get_default_dtype()
    if dtype == torch.float32:
        ref_tensor = REF_DURATIONS_FP32
    else:
        ref_tensor = REF_DURATIONS_FP16

    audio = example_rate16k_audio["waveform"].mean(
        1
    )  # averaging mono to get [Time] shape required by aligner

    extractor = AlignmentExtractor(
        aligner_name,
        unit_extractor_name,
        unit_extractor_output_layer_n,
        unit_extractor_kmeans_uri,
        device=device,
        dtype=dtype,
    )

    alignment_durations, _, _ = extractor.extract_alignment(
        audio, REF_TEXT, plot=False, add_trailing_silence=True
    )

    assert_equal(
        alignment_durations, tensor(ref_tensor, device=device, dtype=torch.int64)
    )
