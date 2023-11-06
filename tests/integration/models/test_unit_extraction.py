# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import tensor
from typing import Final

from fairseq2.typing import Device
from seamless_communication.models.inference import Translator
from seamless_communication.models.unit_extraction import UnitExtractor
from tests.common import assert_equal, device


# fmt: off
REF_ENG_UNITS: Final = [8976, 8299,    0,    0, 9692, 5395,  785,  785, 7805, 6193, 2922, 4806, 3362, 3560, 9007, 8119, 8119,  205, 5424, 5424, 5064, 7421, 6547, 9952, 3728, 8544, 3321, 1093, 1443, 7962, 3978, 9631, 5168, 5491, 9133, 9275, 5912, 8729, 5097, 5495, 1650, 5048, 3752, 6756,  963, 5665, 4191, 5205, 5205, 9568, 5092, 5932, 1190, 9339, 5839, 5839, 6244, 5320, 3454, 5216, 721, 6994, 6513, 7754, 3469,  296, 1849, 3254, 3254, 5042, 5042, 3961, 2079, 1907, 1846,  661, 2225,  944, 9295, 4712, 1785, 6060, 8701, 7646, 1355, 2876, 8199, 5901, 8199, 3861, 5153, 6420, 2897, 1389,  334, 6334]
# fmt: on


def test_unit_extraction() -> None:
    model_name = "seamlessM4T_v2_large"
    english_text = "Hello! I hope you're all doing well."

    if device == Device("cpu"):
        dtype = torch.float32
    else:
        dtype = torch.float16

    translator = Translator(model_name, "vocoder_commercial", device, dtype=dtype)
    unit_extractor = UnitExtractor(
        "xlsr2_1b_v2",
        "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
        device=device,
    )

    # Generate english speech for the english text.
    _, speech_output = translator.predict(
        english_text,
        "t2st",
        "eng",
        src_lang="eng",
    )
    assert speech_output is not None

    units = unit_extractor.predict(speech_output.audio_wavs[0][0], 34)

    assert_equal(units, tensor(REF_ENG_UNITS, device=device, dtype=torch.int64))
