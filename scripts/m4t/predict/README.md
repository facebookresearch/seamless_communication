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
m4t_predict <path_to_input_audio> s2st <tgt_lang> --output_path <path_to_save_audio> --model_name seamlessM4T_large
```

**S2TT**:
```bash
m4t_predict <path_to_input_audio> s2tt <tgt_lang>
```

**T2TT**:
```bash
m4t_predict <input_text> t2tt <tgt_lang> --src_lang <src_lang>
```

**T2ST**:
```bash
m4t_predict <input_text> t2st <tgt_lang> --src_lang <src_lang> --output_path <path_to_save_audio>
```

**ASR**:
```bash
m4t_predict <path_to_input_audio> asr <tgt_lang>
```

Note that it takes 16kHz audio now. Here's how you could resample your audio:
```python
import torchaudio
resample_rate = 16000
waveform, sample_rate = torchaudio.load(<path_to_input_audio>)
resampler = torchaudio.transforms.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
torchaudio.save(<path_to_resampled_audio>, resampled_waveform, resample_rate)
```
## Inference breakdown

Inference calls for the `Translator` object instantiated with a multitask UnitY model with the options:
- [`seamlessM4T_large`](https://huggingface.co/facebook/seamless-m4t-large)
- [`seamlessM4T_medium`](https://huggingface.co/facebook/seamless-m4t-medium)

and a vocoder `vocoder_36langs`

```python
import torch
import torchaudio
from seamless_communication.models.inference import Translator


# Initialize a Translator object with a multitask model, vocoder on the GPU.
translator = Translator("seamlessM4T_large", "vocoder_36langs", torch.device("cuda:0"), torch.float16)
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

## Supported languages
Listed below, are the languages supported by SeamlessM4T models.
The `source` column specifies whether a language is supported as source speech (`Sp`) and/or source text (`Tx`).
The `target` column specifies whether a language is supported as target speech (`Sp`) and/or target text (`Tx`).

| code | language               | script     | Source | Target |
| ---- | ---------------------- | ---------- | ------ | ------ |
| afr  | Afrikaans              | Latn       | Sp, Tx | Tx     |
| amh  | Amharic                | Ethi       | Sp, Tx | Tx     |
| arb  | Modern Standard Arabic | Arab       | Sp, Tx | Sp, Tx |
| ary  | Moroccan Arabic        | Arab       | Sp, Tx | Tx     |
| arz  | Egyptian Arabic        | Arab       | Sp, Tx | Tx     |
| asm  | Assamese               | Beng       | Sp, Tx | Tx     |
| ast  | Asturian               | Latn       | Sp     | \--    |
| azj  | North Azerbaijani      | Latn       | Sp, Tx | Tx     |
| bel  | Belarusian             | Cyrl       | Sp, Tx | Tx     |
| ben  | Bengali                | Beng       | Sp, Tx | Sp, Tx |
| bos  | Bosnian                | Latn       | Sp, Tx | Tx     |
| bul  | Bulgarian              | Cyrl       | Sp, Tx | Tx     |
| cat  | Catalan                | Latn       | Sp, Tx | Sp, Tx |
| ceb  | Cebuano                | Latn       | Sp, Tx | Tx     |
| ces  | Czech                  | Latn       | Sp, Tx | Sp, Tx |
| ckb  | Central Kurdish        | Arab       | Sp, Tx | Tx     |
| cmn  | Mandarin Chinese       | Hans, Hant | Sp, Tx | Sp, Tx |
| cym  | Welsh                  | Latn       | Sp, Tx | Sp, Tx |
| dan  | Danish                 | Latn       | Sp, Tx | Sp, Tx |
| deu  | German                 | Latn       | Sp, Tx | Sp, Tx |
| ell  | Greek                  | Grek       | Sp, Tx | Tx     |
| eng  | English                | Latn       | Sp, Tx | Sp, Tx |
| est  | Estonian               | Latn       | Sp, Tx | Sp, Tx |
| eus  | Basque                 | Latn       | Sp, Tx | Tx     |
| fin  | Finnish                | Latn       | Sp, Tx | Sp, Tx |
| fra  | French                 | Latn       | Sp, Tx | Sp, Tx |
| gaz  | West Central Oromo     | Latn       | Sp, Tx | Tx     |
| gle  | Irish                  | Latn       | Sp, Tx | Tx     |
| glg  | Galician               | Latn       | Sp, Tx | Tx     |
| guj  | Gujarati               | Gujr       | Sp, Tx | Tx     |
| heb  | Hebrew                 | Hebr       | Sp, Tx | Tx     |
| hin  | Hindi                  | Deva       | Sp, Tx | Sp, Tx |
| hrv  | Croatian               | Latn       | Sp, Tx | Tx     |
| hun  | Hungarian              | Latn       | Sp, Tx | Tx     |
| hye  | Armenian               | Armn       | Sp, Tx | Tx     |
| ibo  | Igbo                   | Latn       | Sp, Tx | Tx     |
| ind  | Indonesian             | Latn       | Sp, Tx | Sp, Tx |
| isl  | Icelandic              | Latn       | Sp, Tx | Tx     |
| ita  | Italian                | Latn       | Sp, Tx | Sp, Tx |
| jav  | Javanese               | Latn       | Sp, Tx | Tx     |
| jpn  | Japanese               | Jpan       | Sp, Tx | Sp, Tx |
| kam  | Kamba                  | Latn       | Sp     | \--    |
| kan  | Kannada                | Knda       | Sp, Tx | Tx     |
| kat  | Georgian               | Geor       | Sp, Tx | Tx     |
| kaz  | Kazakh                 | Cyrl       | Sp, Tx | Tx     |
| kea  | Kabuverdianu           | Latn       | Sp     | \--    |
| khk  | Halh Mongolian         | Cyrl       | Sp, Tx | Tx     |
| khm  | Khmer                  | Khmr       | Sp, Tx | Tx     |
| kir  | Kyrgyz                 | Cyrl       | Sp, Tx | Tx     |
| kor  | Korean                 | Kore       | Sp, Tx | Sp, Tx |
| lao  | Lao                    | Laoo       | Sp, Tx | Tx     |
| lit  | Lithuanian             | Latn       | Sp, Tx | Tx     |
| ltz  | Luxembourgish          | Latn       | Sp     | \--    |
| lug  | Ganda                  | Latn       | Sp, Tx | Tx     |
| luo  | Luo                    | Latn       | Sp, Tx | Tx     |
| lvs  | Standard Latvian       | Latn       | Sp, Tx | Tx     |
| mai  | Maithili               | Deva       | Sp, Tx | Tx     |
| mal  | Malayalam              | Mlym       | Sp, Tx | Tx     |
| mar  | Marathi                | Deva       | Sp, Tx | Tx     |
| mkd  | Macedonian             | Cyrl       | Sp, Tx | Tx     |
| mlt  | Maltese                | Latn       | Sp, Tx | Sp, Tx |
| mni  | Meitei                 | Beng       | Sp, Tx | Tx     |
| mya  | Burmese                | Mymr       | Sp, Tx | Tx     |
| nld  | Dutch                  | Latn       | Sp, Tx | Sp, Tx |
| nno  | Norwegian Nynorsk      | Latn       | Sp, Tx | Tx     |
| nob  | Norwegian Bokmål       | Latn       | Sp, Tx | Tx     |
| npi  | Nepali                 | Deva       | Sp, Tx | Tx     |
| nya  | Nyanja                 | Latn       | Sp, Tx | Tx     |
| oci  | Occitan                | Latn       | Sp     | \--    |
| ory  | Odia                   | Orya       | Sp, Tx | Tx     |
| pan  | Punjabi                | Guru       | Sp, Tx | Tx     |
| pbt  | Southern Pashto        | Arab       | Sp, Tx | Tx     |
| pes  | Western Persian        | Arab       | Sp, Tx | Sp, Tx |
| pol  | Polish                 | Latn       | Sp, Tx | Sp, Tx |
| por  | Portuguese             | Latn       | Sp, Tx | Sp, Tx |
| ron  | Romanian               | Latn       | Sp, Tx | Sp, Tx |
| rus  | Russian                | Cyrl       | Sp, Tx | Sp, Tx |
| slk  | Slovak                 | Latn       | Sp, Tx | Sp, Tx |
| slv  | Slovenian              | Latn       | Sp, Tx | Tx     |
| sna  | Shona                  | Latn       | Sp, Tx | Tx     |
| snd  | Sindhi                 | Arab       | Sp, Tx | Tx     |
| som  | Somali                 | Latn       | Sp, Tx | Tx     |
| spa  | Spanish                | Latn       | Sp, Tx | Sp, Tx |
| srp  | Serbian                | Cyrl       | Sp, Tx | Tx     |
| swe  | Swedish                | Latn       | Sp, Tx | Sp, Tx |
| swh  | Swahili                | Latn       | Sp, Tx | Sp, Tx |
| tam  | Tamil                  | Taml       | Sp, Tx | Tx     |
| tel  | Telugu                 | Telu       | Sp, Tx | Sp, Tx |
| tgk  | Tajik                  | Cyrl       | Sp, Tx | Tx     |
| tgl  | Tagalog                | Latn       | Sp, Tx | Sp, Tx |
| tha  | Thai                   | Thai       | Sp, Tx | Sp, Tx |
| tur  | Turkish                | Latn       | Sp, Tx | Sp, Tx |
| ukr  | Ukrainian              | Cyrl       | Sp, Tx | Sp, Tx |
| urd  | Urdu                   | Arab       | Sp, Tx | Sp, Tx |
| uzn  | Northern Uzbek         | Latn       | Sp, Tx | Sp, Tx |
| vie  | Vietnamese             | Latn       | Sp, Tx | Sp, Tx |
| xho  | Xhosa                  | Latn       | Sp     | \--    |
| yor  | Yoruba                 | Latn       | Sp, Tx | Tx     |
| yue  | Cantonese              | Hant       | Sp, Tx | Tx     |
| zlm  | Colloquial Malay       | Latn       | Sp     | \--    |
| zsm  | Standard Malay         | Latn       | Tx     | Tx     |
| zul  | Zulu                   | Latn       | Sp, Tx | Tx     |
