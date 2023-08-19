# On-device Models

Apart from SeamlessM4T-LARGE (2.3B) and SeamlessM4T-MEDIUM (1.2B) models, we are also developing a small model (281M) targeting for on-device inference.
This folder contains an example to run an exported small model covering most tasks (ASR/S2TT/S2ST). The model could be executed on popular mobile devices with Pytorch Mobile (https://pytorch.org/mobile/home/).

## Overview
| Model   | Disk Size | Supported Tasks         | Supported Languages|
|---------|----------------------|-------------------------|-------------------------|
| [UnitY-Small](https://dl.fbaipublicfiles.com/seamless_aug/models/small_unity/unity_on_device.ptl) | 862MB        | S2ST, S2TT, ASR |eng, fra, hin, por, spa|
| [UnitY-Small-S2T](https://dl.fbaipublicfiles.com/seamless_aug/models/small_unity/unity_on_device_s2t.ptl) | 637MB        | S2TT, ASR    |eng, fra,hin,  por, spa|

UnitY-Small-S2T is a pruned version of UnitY-Small without 2nd pass unit decoding.

## Inference
To use exported model, users don't need seamless_communication or fairseq2 dependency.
```
import torchaudio
import torch
audio_input, _ = torchaudio.load(TEST_AUDIO_PATH) # Load waveform using torchaudio

s2t_model = torch.jit.load("unity_on_device_s2t.ptl") # Load exported S2T model
text = s2t_model(audio_input, tgt_lang=TGT_LANG) # Forward call with tgt_lang specified for ASR or S2TT
print(f"{lang}:{text}")

s2st_model = torch.jit.load("unity_on_device.ptl")
text, units, waveform = s2st_model(audio_input, tgt_lang=TGT_LANG) # S2ST model also returns waveform
print(f"{lang}:{text}")
torchaudio.save(f"{OUTPUT_FOLDER}/{lang}.wav", waveform.unsqueeze(0), sample_rate=16000) # Save output waveform to local file
```

Also running the exported model doesn't need python runtime. For example, you could load this model in C++ following [this tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html), or building your own on-device applications similar to [this example](https://github.com/pytorch/ios-demo-app/tree/master/SpeechRecognition)
## Metrics
### S2TT BLEU on FLEURS
Eng-X
| TGT_LANG  | BLEU |
|---------|----------------------|
| fra|?|
| hin|?|
| por|?|
| spa|?|

X-Eng
| SRC_LANG  | BLEU |
|---------|----------------------|
| fra|?|
| hin|?|
| por|?|
| spa|?|

### S2ST BLEU on FLEURS
Eng-X
| TGT_LANG  | BLEU |
|---------|----------------------|
| fra|?|
| hin|?|
| por|?|
| spa|?|

X-Eng
| SRC_LANG  | BLEU |
|---------|----------------------|
| fra|?|
| hin|?|
| por|?|
| spa|?|

### ASR WER on FLEURS
| LANG  | WER |
|---------|----------------------|
| eng|?|
| fra|?|
| hin|?|
| por|?|
| spa|?|
