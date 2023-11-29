# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Final

import torch
from fairseq2.data.audio import AudioDecoderOutput
from torch import tensor

from seamless_communication.inference import Translator
from seamless_communication.inference.pretssel_generator import PretsselGenerator
from seamless_communication.models.unit_extractor import UnitExtractor
from tests.common import (
    assert_unit_close,
    convert_to_collated_fbank,
    device,
)

# fmt: off
REF_UNITS: Final = [8976, 6589, 6589, 5736, 7542, 6515, 1240, 8335, 2381, 1076, 1076, 3380, 4085, 8207, 7957, 4446, 2641, 2544, 5552, 5529, 6319, 2779, 2890, 2890, 3229, 3303, 9751, 1979, 664, 1859, 1302, 528, 1303, 9543, 5770, 3532, 1286, 1286, 1727, 9287, 5248, 5586, 594, 3385, 2613, 1717, 7529, 7634, 931, 1602, 4512, 850, 2748, 5056, 1086, 2320, 2320, 9320, 3223, 5592, 1122, 419, 24, 4126, 5200, 2712, 9549, 8676, 8676, 3443, 7598, 7598, 2200, 2745, 1215, 118, 3840, 2703, 1616, 8788, 1240, 3349, 4890, 2756, 166, 9574, 9773, 5887, 2516, 9332, 6092, 3377, 4334, 3127, 3127, 3127, 944, 3089, 5947, 6572, 6572, 7561, 4358, 4358, 4358, 8124, 5549, 9275, 82, 8830, 8830, 5949, 22, 6729, 6878, 3817, 1871, 6092, 1441, 3127, 3928, 8254, 7984, 1116, 2796, 1806, 3710, 797, 9269, 576, 576, 2020, 137, 6624, 3815, 8690, 3634, 6036, 3530, 8719, 3458, 138, 8745, 5233, 2235, 8580, 8580, 6831, 2709, 7136, 9693, 3437, 3437, 3238, 4368, 2321, 2321, 391, 391, 4976, 8622, 6722, 3864, 9113, 9113, 7222, 7222, 7937, 999, 1286, 1286, 7789, 9396, 9603, 6690, 5233, 2235, 618, 8830, 6954, 3668, 4302, 596, 1934, 2886, 2704, 9097, 4161, 458, 4147, 9245, 9245, 3127, 3127, 944, 9676, 9676, 3468, 270, 270, 4608, 5549, 4182, 102, 8568, 1286, 1286, 5087, 817, 4153, 207, 207, 3763, 6415, 5188, 6010, 554, 753, 9953, 5104, 3828, 1879, 995, 9683, 6932, 3644, 2683, 9335, 183, 5525, 7023, 9568, 6222, 6315, 676, 3443, 6971, 2084, 999, 1286, 1286, 9620, 9620, 1048, 5577, 9328, 4963, 1364, 8328, 4573, 4573, 7917, 7917, 560, 2020, 4923, 137, 9542, 5832, 9775, 4780, 9400, 2745, 2745, 8984, 628, 8834, 6932, 3817, 8312, 5393, 458, 4147, 9191, 2225, 2759, 8980, 2351, 193, 1476, 9347, 3063, 2076, 3641, 1614, 9832, 3554, 8197, 5589, 5589, 7306, 184, 1708, 2954, 2954, 3485, 3485, 7665, 8909, 5405, 3590, 3590, 3446, 6442, 6442, 2802, 5549, 3791]
REF_WAVE_EXTRACTED_UNITS: Final = [8976, 2066, 3800, 2357, 2357, 8080, 9479, 2181, 311, 7241, 5301, 9666, 9925, 940, 9479, 9479, 9479, 3151, 9666, 9925, 2937, 9479, 9479, 3043, 9666, 9189, 9189, 4821, 2937, 2357, 9479, 9479, 9666, 9666, 9666, 9666, 9666, 9666, 9479, 1369, 247, 5025, 5574, 940, 2937, 9479, 9479, 9666, 9666, 9666, 9666, 9666, 5025, 9666, 9666, 9666, 9666, 9666, 9925, 9666, 9479, 9479, 9666, 9666, 9479, 9666, 9479, 9666, 1589, 9666, 9362, 940, 2937, 2937, 9479, 9479, 8063, 9666, 9925, 2937, 9479, 9479, 9666, 9666, 9666, 2130, 4978, 1589, 5574, 5574, 9925, 2937, 9479, 515, 2379, 9666, 9666, 9666, 1589, 4978, 9532, 225, 225, 225, 1251, 225, 3978, 3800, 6343, 1840, 8080, 9666, 9479, 5514, 9666, 6606, 940, 2937, 9479, 9479, 9479, 9666, 9666, 9666, 9666, 9479, 9479, 9666, 9666, 9666, 940, 8080, 9479, 9479, 9479, 9666, 9666, 9666, 9479, 9479, 515, 247, 5025, 5574, 940, 9536, 9479, 9479, 9666, 9666, 9666, 9666, 9666, 1369, 9666, 1653, 4978, 530, 1589, 5574, 940, 940, 9479, 9479, 9479, 9666, 9666, 9925, 9666, 2937, 9479, 8770, 515, 9666, 2130, 5574, 5574, 940, 2937, 9479, 9479, 9479, 8770, 1369, 9580, 1589, 1589, 5574, 940, 2937, 9479, 5634, 9666, 9479, 9202, 1351, 8193, 4660, 4660, 4660, 1463, 1251, 2130, 5574, 1840, 2937, 9479, 9479, 515, 2066, 1653, 7962, 530, 1589, 9666, 940, 940, 9479, 9479, 9666, 9666, 9479, 515, 515, 2720, 8819, 530, 9666, 8063, 940, 2937, 9666, 9666, 9666, 9479, 9666, 9666, 2379, 9925, 2937, 9479, 9479, 1351, 8193, 1589, 9666, 1589, 5574, 940, 2937, 9479, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9666, 9479, 1369, 7810, 9666, 1589, 8063, 5574, 940, 940, 2937, 9479, 9479, 9666, 9479, 311, 1369, 1589, 1589, 5574, 940, 2937, 9479, 7848, 3511, 1589, 1795, 5574, 940, 940, 5786, 2003, 8857, 8193, 8193, 1653, 979, 8471, 8471, 1275, 1885, 225, 225, 4199]
# fmt: on


def test_seamless_expressivity(example_rate16k_audio: AudioDecoderOutput) -> None:
    # this model is seeing non-deterministic behavior (fp32 is better)
    dtype = torch.float32

    audio_dict = example_rate16k_audio

    feat = convert_to_collated_fbank(audio_dict, dtype=dtype)

    unity_model_name = "seamless_expressivity"
    vocoder_model_name = "vocoder_mel"
    pretssel_model_name = "pretssel_v1"
    target_lang = "fra"

    translator = Translator(unity_model_name, None, device, dtype=dtype)

    _, speech_output = translator.predict(
        feat,
        "s2st",
        target_lang,
        prosody_encoder_input=feat,
    )

    assert speech_output is not None

    units = tensor(speech_output.units[0], device=device, dtype=torch.int64)

    # same target units
    assert_unit_close(units, REF_UNITS)

    pretssel_generator = PretsselGenerator(
        unity_model_name,
        vocoder_model_name,
        pretssel_model_name,
        device=device,
        dtype=dtype,
    )

    # same target mel_spectrogram
    speech_output = pretssel_generator.predict(
        speech_output.units,
        tgt_lang=target_lang,
        prosody_encoder_input=feat,
    )

    # UnitExtrator only operates in fp32
    waveform = speech_output.audio_wavs[0][0].float()

    unit_extractor = UnitExtractor(
        "xlsr2_1b_v2",
        "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        device=device,
    )
    units = unit_extractor.predict(waveform, 34)

    assert_unit_close(units, REF_WAVE_EXTRACTED_UNITS)
