# Inference with SeamlessM4T models
Refer to the [SeamlessM4T README](../../../../../docs/m4t) for an overview of the M4T models.

Inference is run with the CLI, from the root directory of the repository.

The model can be specified with `--model_name` `seamlessM4T_v2_large`, `seamlessM4T_large` or `seamlessM4T_medium`:

**S2ST**:
```bash
m4t_predict <path_to_input_audio> --task s2st --tgt_lang <tgt_lang> --output_path <path_to_save_audio> --model_name seamlessM4T_large
```

**S2TT**:
```bash
m4t_predict <path_to_input_audio> --task s2tt --tgt_lang <tgt_lang>
```

**T2TT**:
```bash
m4t_predict <input_text> --task t2tt --tgt_lang <tgt_lang> --src_lang <src_lang>
```

**T2ST**:
```bash
m4t_predict <input_text> --task t2st --tgt_lang <tgt_lang> --src_lang <src_lang> --output_path <path_to_save_audio>
```

**ASR**:
```bash
m4t_predict <path_to_input_audio> --task asr --tgt_lang <tgt_lang>
```
Please set --ngram-filtering to True to get the same translation performance as the [demo](https://seamless.metademolab.com/).

The input audio must be 16kHz currently. Here's how you could resample your audio:
```python
import torchaudio
resample_rate = 16000
waveform, sample_rate = torchaudio.load(<path_to_input_audio>)
resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
torchaudio.save(<path_to_resampled_audio>, resampled_waveform, resample_rate)
```
## Inference breakdown

Inference calls for the `Translator` object instantiated with a multitask UnitY or UnitY2 model with the options:
- [`seamlessM4T_v2_large`](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [`seamlessM4T_large`](https://huggingface.co/facebook/seamless-m4t-large)
- [`seamlessM4T_medium`](https://huggingface.co/facebook/seamless-m4t-medium)

and a vocoder:
- `vocoder_v2` for `seamlessM4T_v2_large`.
- `vocoder_36langs` for `seamlessM4T_large` or `seamlessM4T_medium`.

```python
import torch
import torchaudio
from seamless_communication.inference import Translator


# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:0"), torch.float16)
```

Now `predict()` can be used to run inference as many times on any of the supported tasks.

Given an input audio with `<path_to_input_audio>` or an input text `<input_text>` in `<src_lang>`,
we first set the `text_generation_opts`, `unit_generation_opts` and then translate into `<tgt_lang>` as follows:

## S2ST and T2ST:

```python
# S2ST
text_output, speech_output = translator.predict(
    input=<path_to_input_audio>, 
    task_str="S2ST", 
    tgt_lang=<tgt_lang>, 
    text_generation_opts=text_generation_opts, 
    unit_generation_opts=unit_generation_opts
)

# T2ST
text_output, speech_output = translator.predict(
    input=<input_text>, 
    task_str="T2ST", 
    tgt_lang=<tgt_lang>, 
    src_lang=<src_lang>, 
    text_generation_opts=text_generation_opts,
    unit_generation_opts=unit_generation_opts
)

```
Note that `<src_lang>` must be specified for T2ST.

The generated units are synthesized and the output audio file is saved with:

```python
# Save the translated audio generation.
torchaudio.save(
    <path_to_save_audio>,
    speech_output.audio_wavs[0][0].cpu(),
    sample_rate=speech_output.sample_rate,
)
```
## S2TT, T2TT and ASR:

```python
# S2TT
text_output, _ = translator.predict(
    input=<path_to_input_audio>, 
    task_str="S2TT", 
    tgt_lang=<tgt_lang>, 
    text_generation_opts=text_generation_opts, 
    unit_generation_opts=None
)

# ASR
# This is equivalent to S2TT with `<tgt_lang>=<src_lang>`.
    text_output, _ = translator.predict(
    input=<path_to_input_audio>, 
    task_str="ASR", 
    tgt_lang=<src_lang>, 
    text_generation_opts=text_generation_opts, 
    unit_generation_opts=None
)

# T2TT
text_output, _ = translator.predict(
    input=<input_text>, 
    task_str="T2TT", 
    tgt_lang=<tgt_lang>, 
    src_lang=<src_lang>, 
    text_generation_opts=text_generation_opts, 
    unit_generation_opts=None
)

```
Note that `<src_lang>` must be specified for T2TT
