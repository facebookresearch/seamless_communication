#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import pathlib
import getpass

import gradio as gr
import numpy as np
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from huggingface_hub import snapshot_download
from seamless_communication.inference import Translator

from lang_list import (
    ASR_TARGET_LANGUAGE_NAMES,
    LANGUAGE_NAME_TO_CODE,
    S2ST_TARGET_LANGUAGE_NAMES,
    S2TT_TARGET_LANGUAGE_NAMES,
    T2ST_TARGET_LANGUAGE_NAMES,
    T2TT_TARGET_LANGUAGE_NAMES,
    TEXT_SOURCE_LANGUAGE_NAMES,
)

user = getpass.getuser() # this is not portable on windows
CHECKPOINTS_PATH = pathlib.Path(os.getenv("CHECKPOINTS_PATH", f"/home/{user}/app/models"))
if not CHECKPOINTS_PATH.exists():
    snapshot_download(repo_id="facebook/seamless-m4t-v2-large", repo_type="model", local_dir=CHECKPOINTS_PATH)
asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")
demo_metadata = [
    {
        "name": "seamlessM4T_v2_large@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/seamlessM4T_v2_large.pt",
        "char_tokenizer": f"file://{CHECKPOINTS_PATH}/spm_char_lang38_tc.model",
    },
    {
        "name": "vocoder_v2@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/vocoder_v2.pt",
    },
]
asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))

DESCRIPTION = """\
# SeamlessM4T
[SeamlessM4T](https://github.com/facebookresearch/seamless_communication) is designed to provide high-quality
translation, allowing people from different linguistic communities to communicate effortlessly through speech and text.
This unified model enables multiple tasks like Speech-to-Speech (S2ST), Speech-to-Text (S2TT), Text-to-Speech (T2ST)
translation and more, without relying on multiple separate models.
"""

CACHE_EXAMPLES = os.getenv("CACHE_EXAMPLES") == "1" and torch.cuda.is_available()

AUDIO_SAMPLE_RATE = 16000.0
MAX_INPUT_AUDIO_LENGTH = 60  # in seconds
DEFAULT_TARGET_LANGUAGE = "French"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

translator = Translator(
    model_name_or_card="seamlessM4T_v2_large",
    vocoder_name_or_card="vocoder_v2",
    device=device,
    dtype=dtype,
    apply_mintox=True,
)


def preprocess_audio(input_audio: str) -> None:
    arr, org_sr = torchaudio.load(input_audio)
    new_arr = torchaudio.functional.resample(arr, orig_freq=org_sr, new_freq=AUDIO_SAMPLE_RATE)
    max_length = int(MAX_INPUT_AUDIO_LENGTH * AUDIO_SAMPLE_RATE)
    if new_arr.shape[1] > max_length:
        new_arr = new_arr[:, :max_length]
        gr.Warning(f"Input audio is too long. Only the first {MAX_INPUT_AUDIO_LENGTH} seconds is used.")
    torchaudio.save(input_audio, new_arr, sample_rate=int(AUDIO_SAMPLE_RATE))


def run_s2st(
    input_audio: str, source_language: str, target_language: str
) -> tuple[tuple[int, np.ndarray] | None, str]:
    preprocess_audio(input_audio)
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, out_audios = translator.predict(
        input=input_audio,
        task_str="S2ST",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )
    out_text = str(out_texts[0])
    out_wav = out_audios.audio_wavs[0].cpu().detach().numpy()
    return (int(AUDIO_SAMPLE_RATE), out_wav), out_text


def run_s2tt(input_audio: str, source_language: str, target_language: str) -> str:
    preprocess_audio(input_audio)
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="S2TT",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )
    return str(out_texts[0])


def run_t2st(input_text: str, source_language: str, target_language: str) -> tuple[tuple[int, np.ndarray] | None, str]:
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, out_audios = translator.predict(
        input=input_text,
        task_str="T2ST",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )
    out_text = str(out_texts[0])
    out_wav = out_audios.audio_wavs[0].cpu().detach().numpy()
    return (int(AUDIO_SAMPLE_RATE), out_wav), out_text


def run_t2tt(input_text: str, source_language: str, target_language: str) -> str:
    source_language_code = LANGUAGE_NAME_TO_CODE[source_language]
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, _ = translator.predict(
        input=input_text,
        task_str="T2TT",
        src_lang=source_language_code,
        tgt_lang=target_language_code,
    )
    return str(out_texts[0])


def run_asr(input_audio: str, target_language: str) -> str:
    preprocess_audio(input_audio)
    target_language_code = LANGUAGE_NAME_TO_CODE[target_language]
    out_texts, _ = translator.predict(
        input=input_audio,
        task_str="ASR",
        src_lang=target_language_code,
        tgt_lang=target_language_code,
    )
    return str(out_texts[0])


with gr.Blocks() as demo_s2st:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech", type="filepath")
                source_language = gr.Dropdown(
                    label="Source language",
                    choices=ASR_TARGET_LANGUAGE_NAMES,
                    value="English",
                )
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=S2ST_TARGET_LANGUAGE_NAMES,
                    value=DEFAULT_TARGET_LANGUAGE,
                )
            btn = gr.Button("Translate")
        with gr.Column():
            with gr.Group():
                output_audio = gr.Audio(
                    label="Translated speech",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
                output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_audio, source_language, target_language],
        outputs=[output_audio, output_text],
        fn=run_s2st,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    btn.click(
        fn=run_s2st,
        inputs=[input_audio, source_language, target_language],
        outputs=[output_audio, output_text],
        api_name="s2st",
    )

with gr.Blocks() as demo_s2tt:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech", type="filepath")
                source_language = gr.Dropdown(
                    label="Source language",
                    choices=ASR_TARGET_LANGUAGE_NAMES,
                    value="English",
                )
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=S2TT_TARGET_LANGUAGE_NAMES,
                    value=DEFAULT_TARGET_LANGUAGE,
                )
            btn = gr.Button("Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_audio, source_language, target_language],
        outputs=output_text,
        fn=run_s2tt,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    btn.click(
        fn=run_s2tt,
        inputs=[input_audio, source_language, target_language],
        outputs=output_text,
        api_name="s2tt",
    )

with gr.Blocks() as demo_t2st:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_text = gr.Textbox(label="Input text")
                with gr.Row():
                    source_language = gr.Dropdown(
                        label="Source language",
                        choices=TEXT_SOURCE_LANGUAGE_NAMES,
                        value="English",
                    )
                    target_language = gr.Dropdown(
                        label="Target language",
                        choices=T2ST_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    )
            btn = gr.Button("Translate")
        with gr.Column():
            with gr.Group():
                output_audio = gr.Audio(
                    label="Translated speech",
                    autoplay=False,
                    streaming=False,
                    type="numpy",
                )
                output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_text, source_language, target_language],
        outputs=[output_audio, output_text],
        fn=run_t2st,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    gr.on(
        triggers=[input_text.submit, btn.click],
        fn=run_t2st,
        inputs=[input_text, source_language, target_language],
        outputs=[output_audio, output_text],
        api_name="t2st",
    )

with gr.Blocks() as demo_t2tt:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_text = gr.Textbox(label="Input text")
                with gr.Row():
                    source_language = gr.Dropdown(
                        label="Source language",
                        choices=TEXT_SOURCE_LANGUAGE_NAMES,
                        value="English",
                    )
                    target_language = gr.Dropdown(
                        label="Target language",
                        choices=T2TT_TARGET_LANGUAGE_NAMES,
                        value=DEFAULT_TARGET_LANGUAGE,
                    )
            btn = gr.Button("Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_text, source_language, target_language],
        outputs=output_text,
        fn=run_t2tt,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    gr.on(
        triggers=[input_text.submit, btn.click],
        fn=run_t2tt,
        inputs=[input_text, source_language, target_language],
        outputs=output_text,
        api_name="t2tt",
    )

with gr.Blocks() as demo_asr:
    with gr.Row():
        with gr.Column():
            with gr.Group():
                input_audio = gr.Audio(label="Input speech", type="filepath")
                target_language = gr.Dropdown(
                    label="Target language",
                    choices=ASR_TARGET_LANGUAGE_NAMES,
                    value=DEFAULT_TARGET_LANGUAGE,
                )
            btn = gr.Button("Translate")
        with gr.Column():
            output_text = gr.Textbox(label="Translated text")

    gr.Examples(
        examples=[],
        inputs=[input_audio, target_language],
        outputs=output_text,
        fn=run_asr,
        cache_examples=CACHE_EXAMPLES,
        api_name=False,
    )

    btn.click(
        fn=run_asr,
        inputs=[input_audio, target_language],
        outputs=output_text,
        api_name="asr",
    )


with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )

    with gr.Tabs():
        with gr.Tab(label="S2ST"):
            demo_s2st.render()
        with gr.Tab(label="S2TT"):
            demo_s2tt.render()
        with gr.Tab(label="T2ST"):
            demo_t2st.render()
        with gr.Tab(label="T2TT"):
            demo_t2tt.render()
        with gr.Tab(label="ASR"):
            demo_asr.render()


if __name__ == "__main__":
    demo.queue(max_size=50).launch()
