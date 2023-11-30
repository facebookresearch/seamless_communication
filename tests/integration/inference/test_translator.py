# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

from typing import Final

from seamless_communication.inference import Translator
from tests.common import device, get_default_dtype

# fmt: off
ENG_SENTENCE:     Final = "On Monday, scientists from the Stanford University School of Medicine announced the invention of a new diagnostic tool that can sort cells by type: a tiny printable chip that can be manufactured using standard inkjet printers for possibly about one U.S. cent each."
DEU_SENTENCE:     Final = "Am Montag kündigten Wissenschaftler der Stanford University School of Medicine die Erfindung eines neuen Diagnosewerkzeugs an, das Zellen nach Typ sortieren kann: ein winziger druckbarer Chip, der mit Standard-Tintenstrahldruckern für etwa einen US-Cent hergestellt werden kann."
DEU_SENTENCE_V2:  Final = "Am Montag kündigten Wissenschaftler der Stanford University School of Medicine die Erfindung eines neuen diagnostischen Werkzeugs an, das Zellen nach Typ sortieren kann: ein winziger druckbarer Chip, der mit Standard-Tintenstrahldrucker für möglicherweise etwa einen US-Cent pro Stück hergestellt werden kann."
# fmt: on


def test_seamless_m4t_large_t2tt() -> None:
    model_name = "seamlessM4T_large"
    src_lang = "eng"
    tgt_lang = "deu"

    dtype = get_default_dtype()

    translator = Translator(model_name, "vocoder_36langs", device, dtype=dtype)
    text_output, _ = translator.predict(
        ENG_SENTENCE,
        "t2tt",
        tgt_lang,
        src_lang=src_lang,
    )
    assert text_output[0] == DEU_SENTENCE, f"'{text_output[0]}' is not '{DEU_SENTENCE}'"


def test_seamless_m4t_v2_large_t2tt() -> None:
    model_name = "seamlessM4T_v2_large"
    src_lang = "eng"
    tgt_lang = "deu"

    dtype = get_default_dtype()

    translator = Translator(model_name, "vocoder_v2", device, dtype=dtype)
    text_output, _ = translator.predict(
        ENG_SENTENCE,
        "t2tt",
        tgt_lang,
        src_lang=src_lang,
    )
    assert (
        text_output[0] == DEU_SENTENCE_V2
    ), f"'{text_output[0]}' is not '{DEU_SENTENCE_V2}'"


def test_seamless_m4t_v2_large_multiple_tasks() -> None:
    model_name = "seamlessM4T_v2_large"
    english_text = "Hello! I hope you're all doing well."
    ref_spanish_text = "Hola, espero que todo se esté haciendo bien."
    ref_spanish_asr_text = "Hola, espero que todo se esté haciendo bien."

    dtype = get_default_dtype()

    translator = Translator(model_name, "vocoder_v2", device, dtype=dtype)

    # Generate english speech for the english text.
    _, english_speech_output = translator.predict(
        english_text,
        "t2st",
        "eng",
        src_lang="eng",
    )
    assert english_speech_output is not None

    # Translate english speech to spanish speech.
    spanish_text_output, spanish_speech_output = translator.predict(
        english_speech_output.audio_wavs[0][0],
        "s2st",
        "spa",
    )
    assert spanish_speech_output is not None
    assert (
        spanish_text_output[0] == ref_spanish_text
    ), f"'{spanish_text_output[0]}' is not '{ref_spanish_text}'"

    # Run ASR on the spanish speech.
    spanish_asr_text_output, _ = translator.predict(
        spanish_speech_output.audio_wavs[0][0],
        "asr",
        "spa",
    )
    assert (
        spanish_asr_text_output[0] == ref_spanish_asr_text
    ), f"{spanish_asr_text_output[0]} is not {ref_spanish_asr_text}'"
