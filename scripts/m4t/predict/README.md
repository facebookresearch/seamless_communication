# Inference with SeamlessM4T models

SeamlessM4T models currently support five tasks:
- Speech-to-speech translation (S2ST)
- Speech-to-text translation (S2TT)
- Text-to-speech translation (T2ST)
- Text-to-text translation (T2TT)
- Automatic speech recognition (ASR)



## Quick start:
Inference is run with the CLI, from the root directory of the repository.

The model can be specified with `--model_name` `seamlessM4T_large` or `seamlessM4T_medium`:

**S2ST**:
```bash
python scripts/m4t/predict/predict.py <path_to_input_audio> s2st <tgt_lang> --output_path <path_to_save_audio> --model_name seamlessM4T_large
```

**S2TT**:
```bash
python scripts/m4t/predict/predict.py <path_to_input_audio> s2tt <tgt_lang>
```

**T2TT**:
```bash
python scripts/m4t/predict/predict.py <input_text> t2tt <tgt_lang> --src_lang <src_lang>
```

**T2ST**:
```bash
python scripts/m4t/predict/predict.py <input_text> t2st <tgt_lang> --src_lang <src_lang> --output_path <path_to_save_audio>
```

**ASR**:
```bash
python scripts/m4t/predict/predict.py <path_to_input_audio> asr <tgt_lang>
```

## Inference breakdown

Inference calls for the `Translator` object instantiated with a Multitasking UnitY model with the options:
- `seamlessM4T_large`
- `seamlessM4T_medium`

and a vocoder `vocoder_36langs`

```python
import torch
import torchaudio
from seamless_communication.models.inference import Translator


# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:0"))
```

Now `predict()` can be used to run inference as many times on any of the supported tasks.

Given an input audio with `<path_to_input_audio>` or an input text `<input_text>` in `<src_lang>`,
we can translate into `<tgt_lang>` as follows:

## S2ST and T2ST:

```python
# S2ST
translated_text, wav, sr = translator.predict(<path_to_input_audio>, "s2st", <tgt_lang>)

# T2ST
translated_text, wav, sr = translator.predict(<input_text>, "t2st", <tgt_lang>, src_lang=<src_lang>)

```
Note that `<src_lang>` must be specified for T2ST.

The generated units are synthesized and the output audio file is saved with:

```python
wav, sr = translator.synthesize_speech(<speech_units>, <tgt_lang>)

# Save the translated audio generation.
torchaudio.save(
    <path_to_save_audio>,
    wav[0].cpu(),
    sample_rate=sr,
)
```
## S2TT, T2TT and ASR:

```python
# S2TT
translated_text, _, _ = translator.predict(<path_to_input_audio>, "s2tt", <tgt_lang>)

# ASR
# This is equivalent to S2TT with `<tgt_lang>=<src_lang>`.
transcribed_text, _, _ = translator.predict(<path_to_input_audio>, "asr", <src_lang>)

# T2TT
translated_text, _, _ = translator.predict(<input_text>, "t2tt", <tgt_lang>, src_lang=<src_lang>)

```
Note that `<src_lang>` must be specified for T2TT