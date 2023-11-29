# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import Namespace
from pathlib import Path
from typing import Final, List, Optional, cast
import pytest

import torch
from fairseq2.data import SequenceData, VocabularyInfo
from fairseq2.data.audio import AudioDecoderOutput
from fairseq2.typing import Device
from torch.nn import Module

from seamless_communication.inference import Translator
from seamless_communication.inference.pretssel_generator import PretsselGenerator
from seamless_communication.cli.expressivity.evaluate.pretssel_inference_helper import (
    PretsselGenerator as WatermarkedPretsselGenerator,
)
from seamless_communication.cli.expressivity.evaluate.pretssel_inference import (
    build_data_pipeline,
)
from seamless_communication.models.unity import load_gcmvn_stats
from tests.common import assert_close, convert_to_collated_fbank

N_MEL_BINS = 80
WM_WEIGHT = 0.8

# fmt: off
REF_FRA_UNITS: Final = [8976, 6589, 6589, 5736, 7542, 6515, 1240, 8335, 2381, 1076, 1076, 3380, 4085, 8207, 7957, 4446, 2641, 2544, 5552, 5529, 6319, 2779, 2890, 2890, 3229, 3303, 9751, 1979, 664, 1859, 1302, 528, 1303, 9543, 5770, 3532, 1286, 1286, 1727, 9287, 5248, 5586, 594, 3385, 2613, 1717, 7529, 7634, 931, 1602, 4512, 850, 2748, 5056, 1086, 2320, 2320, 9320, 3223, 5592, 1122, 419, 24, 4126, 5200, 2712, 9549, 8676, 8676, 3443, 7598, 7598, 2200, 2745, 1215, 118, 3840, 2703, 1616, 8788, 1240, 3349, 4890, 2756, 166, 9574, 9773, 5887, 2516, 9332, 6092, 3377, 4334, 3127, 3127, 3127, 944, 3089, 5947, 6572, 6572, 7561, 4358, 4358, 4358, 8124, 5549, 9275, 82, 8830, 8830, 5949, 22, 6729, 6878, 3817, 1871, 6092, 1441, 3127, 3928, 8254, 7984, 1116, 2796, 1806, 3710, 797, 9269, 576, 576, 2020, 137, 6624, 3815, 8690, 3634, 6036, 3530, 8719, 3458, 138, 8745, 5233, 2235, 8580, 8580, 6831, 2709, 7136, 9693, 3437, 3437, 3238, 4368, 2321, 2321, 391, 391, 4976, 8622, 6722, 3864, 9113, 9113, 7222, 7222, 7937, 999, 1286, 1286, 7789, 9396, 9603, 6690, 5233, 2235, 618, 8830, 6954, 3668, 4302, 596, 1934, 2886, 2704, 9097, 4161, 458, 4147, 9245, 9245, 3127, 3127, 944, 9676, 9676, 3468, 270, 270, 4608, 5549, 4182, 102, 8568, 1286, 1286, 5087, 817, 4153, 207, 207, 3763, 6415, 5188, 6010, 554, 753, 9953, 5104, 3828, 1879, 995, 9683, 6932, 3644, 2683, 9335, 183, 5525, 7023, 9568, 6222, 6315, 676, 3443, 6971, 2084, 999, 1286, 1286, 9620, 9620, 1048, 5577, 9328, 4963, 1364, 8328, 4573, 4573, 7917, 7917, 560, 2020, 4923, 137, 9542, 5832, 9775, 4780, 9400, 2745, 2745, 8984, 628, 8834, 6932, 3817, 8312, 5393, 458, 4147, 9191, 2225, 2759, 8980, 2351, 193, 1476, 9347, 3063, 2076, 3641, 1614, 9832, 3554, 8197, 5589, 5589, 7306, 184, 1708, 2954, 2954, 3485, 3485, 7665, 8909, 5405, 3590, 3590, 3446, 6442, 6442, 2802, 5549, 3791]
# fmt: on


def load_watermarking_model() -> Optional[Module]:
    import importlib.util

    # Run in CPU mode until pretssel inconsistent behavious is fixed
    device = Device("cpu")
    dtype = torch.float32
    wm_py_file = Path(__file__).parents[3] / "scripts/watermarking/watermarking.py"
    assert wm_py_file.is_file()
    wm_spec = importlib.util.spec_from_file_location("watermark.f1", wm_py_file)
    assert wm_spec, f"Module not found: {wm_py_file}"
    wm_py_module = importlib.util.module_from_spec(wm_spec)
    assert wm_py_module, f"Invalid Python module file: {wm_py_file}"
    sys.modules["watermark.f1"] = wm_py_module
    assert wm_spec.loader, f"Module cannot be loaded from {wm_py_file}"
    wm_spec.loader.exec_module(wm_py_module)

    return cast(Module, wm_py_module.model_from_checkpoint(device=device, dtype=dtype))


@pytest.mark.parametrize("sr", [16_000, 24_000])
def test_pretssel_vocoder_watermarking(
    example_rate16k_audio: AudioDecoderOutput, sr: int
) -> None:
    """
    Test that the watermarked pretssel vocoder generates the same output
    as the non-watermarked (pretssel_generator)
    """
    # Run in CPU mode until pretssel inconsistent behavious is fixed
    device = Device("cpu")
    dtype = torch.float32

    audio = example_rate16k_audio
    audio["waveform"] = audio["waveform"].to(device, dtype=dtype)
    feat = convert_to_collated_fbank(audio, dtype=dtype)["seqs"][0]
    tgt_lang = "fra"

    feat = feat.to(device, dtype=dtype)

    gcmvn_mean, gcmvn_std = load_gcmvn_stats("pretssel_v1")
    gcmvn_mean = torch.tensor(gcmvn_mean, device=device, dtype=dtype)  # type: ignore[assignment]
    gcmvn_std = torch.tensor(gcmvn_std, device=device, dtype=dtype)  # type: ignore[assignment]

    if sr == 16_000:
        vocoder_model_name = "vocoder_mel"
        pretssel_vocoder_model_name = "vocoder_pretssel_16khz"
    else:
        vocoder_model_name = "vocoder_mel_24khz"
        pretssel_vocoder_model_name = "vocoder_pretssel"

    # non-watermarked vocoder using pretssel generator in inference
    generator = PretsselGenerator(
        "seamless_expressivity",
        vocoder_model_name,
        "pretssel_v1",
        gcmvn_mean=gcmvn_mean,  # type: ignore[arg-type]
        gcmvn_std=gcmvn_std,  # type: ignore[arg-type]
        device=device,
        dtype=dtype,
    )

    # watermarked vocoder using pretssel generator in the evaluation
    vocab_info = VocabularyInfo(size=10004, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1)
    wm_generator = WatermarkedPretsselGenerator(
        pretssel_vocoder_model_name,
        vocab_info=vocab_info,
        device=device,
        dtype=dtype,
    )

    unit_list: List[List[int]] = [REF_FRA_UNITS]
    prosody_input_seqs = SequenceData(
        is_ragged=False,
        seqs=feat.unsqueeze(0),  # add batch dim
        seq_lens=torch.tensor([feat.size(0)]),
    )

    # Run the non-watermark vocoder, followed by a watermarker
    speech_output = generator.predict(
        unit_list,
        tgt_lang=tgt_lang,
        prosody_encoder_input=prosody_input_seqs,
    )
    wav = speech_output.audio_wavs[0].unsqueeze(0)

    watermarker = load_watermarking_model()
    wm = watermarker.get_watermark(wav)  # type: ignore
    wav_wm_hat = wav + WM_WEIGHT * wm

    # Run the watermarked vocoder
    wm_speech_output = wm_generator.predict(
        unit_list,
        tgt_lang=tgt_lang,
        prosody_encoder_input=prosody_input_seqs,
    )
    wav_wm = wm_speech_output.audio_wavs[0]

    # Test that the watermark is detectable
    detection = watermarker.detect_watermark(wav_wm)  # type: ignore
    assert torch.all(detection[:, 1, :] > 0.5)

    # Remove the batch and compare parity on the overlapping frames
    wav_wm = wav_wm.squeeze(0)
    wav_wm_hat = wav_wm_hat.squeeze(0)

    nframes = min(wav_wm_hat.size(1), wav_wm.size(1))
    assert_close(
        wav_wm[:, :nframes],
        wav_wm_hat[:, :nframes],
        atol=0.0,
        rtol=5.0,
    )


@pytest.mark.skip(reason="Skip this test since it's extremely slow.")
def test_e2e_watermark_audio() -> None:
    data_file = "/large_experiments/seamless/data/expressivity/fairseq_manifest/benchmark_20231025/test_examples_20231122.tsv"
    model_name = "seamless_expressivity"

    # Run in CPU mode until pretssel inconsistent behavious is fixed
    device = Device("cpu")
    dtype = torch.float32

    gcmvn_mean, gcmvn_std = load_gcmvn_stats("pretssel_v1")
    gcmvn_mean = torch.tensor(gcmvn_mean, device=device, dtype=dtype)  # type: ignore[assignment]
    gcmvn_std = torch.tensor(gcmvn_std, device=device, dtype=dtype)  # type: ignore[assignment]

    args = Namespace(data_file=data_file, audio_root_dir="", batch_size=4)
    pipeline = build_data_pipeline(
        args, device=device, dtype=dtype, gcmvn_mean=gcmvn_mean, gcmvn_std=gcmvn_std  # type: ignore[arg-type]
    )
    translator = Translator(model_name, None, device=device, dtype=dtype)

    # no watermark
    generator = PretsselGenerator(
        "seamless_expressivity",
        "vocoder_mel_24khz",
        "pretssel_v1",
        gcmvn_mean=gcmvn_mean,  # type: ignore[arg-type]
        gcmvn_std=gcmvn_std,  # type: ignore[arg-type]
        device=device,
        dtype=dtype,
    )
    watermarker = load_watermarking_model()

    # watermark
    vocab_info = VocabularyInfo(size=10004, unk_idx=3, bos_idx=0, eos_idx=2, pad_idx=1)

    wm_generator = WatermarkedPretsselGenerator(
        "vocoder_pretssel",
        vocab_info=vocab_info,
        device=device,
        dtype=dtype,
    )

    sample_id = 0
    for batch in pipeline:
        feat = batch["audio"]["data"]["fbank"]
        prosody_encoder_input = batch["audio"]["data"]["gcmvn_fbank"]

        text_output, unit_out = translator.predict(
            feat,
            task_str="s2st",
            tgt_lang="spa",
            prosody_encoder_input=prosody_encoder_input,
        )
        assert unit_out, "empty translation output"

        speech_out = generator.predict(
            units=unit_out.units,
            tgt_lang="spa",
            prosody_encoder_input=prosody_encoder_input,
        )

        wm_speech_out = wm_generator.predict(
            units=unit_out.units,
            tgt_lang="spa",
            prosody_encoder_input=prosody_encoder_input,
        )

        for i in range(len(text_output)):
            wav_wm = wm_speech_out.audio_wavs[i].squeeze(0)
            wav = speech_out.audio_wavs[i].unsqueeze(0)
            wm = watermarker.get_watermark(wav)  # type: ignore
            wav_wm_hat = wav + 0.8 * wm
            wav_wm_hat = wav_wm_hat.squeeze(0)
            assert_close(wav_wm, wav_wm_hat, atol=0.01, rtol=5.0)
            sample_id += 1
