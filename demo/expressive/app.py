#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import os
import pathlib
import tempfile

import gradio as gr
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from fairseq2.data import Collater
from fairseq2.data.audio import (
    AudioDecoder,
    WaveformToFbankConverter,
    WaveformToFbankOutput,
)

from seamless_communication.inference import SequenceGeneratorOptions
from fairseq2.generation import NGramRepeatBlockProcessor
from fairseq2.memory import MemoryBlock
from huggingface_hub import snapshot_download
from seamless_communication.inference import Translator, SequenceGeneratorOptions
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)
from seamless_communication.cli.expressivity.predict.pretssel_generator import PretsselGenerator

from typing import Tuple
from utils import LANGUAGE_CODE_TO_NAME

DESCRIPTION = """\
# Seamless Expressive
[SeamlessExpressive](https://github.com/facebookresearch/seamless_communication) is a speech-to-speech translation model that captures certain underexplored aspects of prosody such as speech rate and pauses, while preserving the style of one's voice and high content translation quality.
"""

CACHE_EXAMPLES = os.getenv("CACHE_EXAMPLES") == "1" and torch.cuda.is_available()

CHECKPOINTS_PATH = pathlib.Path(os.getenv("CHECKPOINTS_PATH", "/home/user/app/models"))
if not CHECKPOINTS_PATH.exists():
    snapshot_download(repo_id="facebook/seamless-expressive", repo_type="model", local_dir=CHECKPOINTS_PATH)
    snapshot_download(repo_id="facebook/seamless-m4t-v2-large", repo_type="model", local_dir=CHECKPOINTS_PATH)

# Ensure that we do not have any other environment resolvers and always return
# "demo" for demo purposes.
asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")

# Construct an `InProcAssetMetadataProvider` with environment-specific metadata
# that just overrides the regular metadata for "demo" environment. Note the "@demo" suffix.
demo_metadata = [
    {
        "name": "seamless_expressivity@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/m2m_expressive_unity.pt",
        "char_tokenizer": f"file://{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
    {
        "name": "vocoder_pretssel@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/pretssel_melhifigan_wm-final.pt",
    },
    {
        "name": "seamlessM4T_v2_large@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/seamlessM4T_v2_large.pt",
        "char_tokenizer": f"file://{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
]

asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))

LANGUAGE_NAME_TO_CODE = {v: k for k, v in LANGUAGE_CODE_TO_NAME.items()}


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32


MODEL_NAME = "seamless_expressivity"
VOCODER_NAME = "vocoder_pretssel"

# used for ASR for toxicity
m4t_translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card=None,
    device=device,
    dtype=dtype,
)
unit_tokenizer = load_unity_unit_tokenizer(MODEL_NAME)

_gcmvn_mean, _gcmvn_std = load_gcmvn_stats(VOCODER_NAME)
gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)
gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

translator = Translator(
    MODEL_NAME,
    vocoder_name_or_card=None,
    device=device,
    dtype=dtype,
    apply_mintox=False,
)

text_generation_opts = SequenceGeneratorOptions(
    beam_size=5,
    unk_penalty=torch.inf,
    soft_max_seq_len=(0, 200),
    step_processor=NGramRepeatBlockProcessor(
        ngram_size=10,
    ),
)
m4t_text_generation_opts = SequenceGeneratorOptions(
    beam_size=5,
    unk_penalty=torch.inf,
    soft_max_seq_len=(1, 200),
    step_processor=NGramRepeatBlockProcessor(
        ngram_size=10,
    ),
)

pretssel_generator = PretsselGenerator(
    VOCODER_NAME,
    vocab_info=unit_tokenizer.vocab_info,
    device=device,
    dtype=dtype,
)

decode_audio = AudioDecoder(dtype=torch.float32, device=device)

convert_to_fbank = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=False,
    device=device,
    dtype=dtype,
)


def normalize_fbank(data: WaveformToFbankOutput) -> WaveformToFbankOutput:
    fbank = data["fbank"]
    std, mean = torch.std_mean(fbank, dim=0)
    data["fbank"] = fbank.subtract(mean).divide(std)
    data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
    return data


collate = Collater(pad_value=0, pad_to_multiple=1)


AUDIO_SAMPLE_RATE = 16000
MAX_INPUT_AUDIO_LENGTH = 10  # in seconds


def remove_prosody_tokens_from_text(text):
    # filter out prosody tokens, there is only emphasis '*', and pause '='
    text = text.replace("*", "").replace("=", "")
    text = " ".join(text.split())
    return text


def preprocess_audio(input_audio_path: str) -> None:
    arr, org_sr = torchaudio.load(input_audio_path)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio_path, new_arr, sample_rate=AUDIO_SAMPLE_RATE)


def run(
    input_audio_path: str,
    source_language: str,
    target_language: str,
) -> Tuple[str, str]:
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]

    preprocess_audio(input_audio_path)

    with pathlib.Path(input_audio_path).open("rb") as fb:
        block = MemoryBlock(fb.read())
        example = decode_audio(block)

    example = convert_to_fbank(example)
    example = normalize_fbank(example)
    example = collate(example)

    # get transcription for mintox
    source_sentences, _ = m4t_translator.predict(
        input=example["fbank"],
        task_str="S2TT",  # get source text
        tgt_lang=source_language_code,
        text_generation_opts=m4t_text_generation_opts,
    )
    source_text = str(source_sentences[0])

    prosody_encoder_input = example["gcmvn_fbank"]
    text_output, unit_output = translator.predict(
        example["fbank"],
        "S2ST",
        tgt_lang=target_language_code,
        src_lang=source_language_code,
        text_generation_opts=text_generation_opts,
        unit_generation_ngram_filtering=False,
        duration_factor=1.0,
        prosody_encoder_input=prosody_encoder_input,
        src_text=source_text,  # for mintox check
    )
    speech_output = pretssel_generator.predict(
        unit_output.units,
        tgt_lang=target_language_code,
        prosody_encoder_input=prosody_encoder_input,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        torchaudio.save(
            f.name,
            speech_output.audio_wavs[0][0].to(torch.float32).cpu(),
            sample_rate=speech_output.sample_rate,
        )

    text_out = remove_prosody_tokens_from_text(str(text_output[0]))

    return f.name, text_out


TARGET_LANGUAGE_NAMES = [
    "English",
    "French",
    "German",
    "Spanish",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech", type="filepath")
                source_language = gr.Dropdown(
                    label="Source language",
                    choices=TARGET_LANGUAGE_NAMES,
                    value="English",
                )
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=TARGET_LANGUAGE_NAMES,
                    value="French",
                )
            btn = gr.Button()
        with gr.Column():
            with gr.Group():
                output_audio = gr.Audio(label="Translated speech")
                output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_audio, source_language, target_language],
        outputs=[output_audio, output_text],
        fn=run,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    btn.click(
        fn=run,
        inputs=[input_audio, source_language, target_language],
        outputs=[output_audio, output_text],
        api_name="run",
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch()