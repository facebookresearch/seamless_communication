# On-device Models [Experimental]

Apart from SeamlessM4T-LARGE (2.3B) and SeamlessM4T-MEDIUM (1.2B) models, we are also developing a small model (281M) targeting for on-device inference.
This folder contains an example to run an exported small model covering most tasks (ASR/S2TT/S2ST). The model could be executed on popular mobile devices with Pytorch Mobile (https://pytorch.org/mobile/home/).

## Updates
[2023/8/23] Uploaded new on-device models with several fixes to reduce size and avoid OOM. Metrics should be close to what's reported below, will rerun eval and update.   

## Overview
| Model           | Model Card                                                                                   | Checkpoint                                                                                                      | Num Params | Disk Size | Supported Tasks | Supported Languages     |
|-----------------|----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|------------|-----------|-----------------|-------------------------|
| UnitY-Small     | [seamless-m4t-unity-small](https://huggingface.co/facebook/seamless-m4t-unity-small)         | [checkpoint](https://huggingface.co/facebook/seamless-m4t-unity-small/resolve/main/unity_on_device.ptl)         | 281M       | 747MB     | S2ST, S2TT, ASR | eng, fra, hin, por, spa |
| UnitY-Small-S2T | [seamless-m4t-unity-small-s2t](https://huggingface.co/facebook/seamless-m4t-unity-small-s2t) | [checkpoint](https://huggingface.co/facebook/seamless-m4t-unity-small-s2t/resolve/main/unity_on_device_s2t.ptl) | 235M       | 481MB     | S2TT, ASR       | eng, fra, hin, por, spa |

UnitY-Small-S2T is a pruned version of UnitY-Small without 2nd pass unit decoding.

## Inference
To use exported model, users don't need seamless_communication or fairseq2 dependency.
```python
import torchaudio
import torch
audio_input, _ = torchaudio.load(TEST_AUDIO_PATH) # Load waveform using torchaudio

s2t_model = torch.jit.load("unity_on_device_s2t.ptl") # Load exported S2T model
with torch.no_grad():
    text = s2t_model(audio_input, tgt_lang=TGT_LANG) # Forward call with tgt_lang specified for ASR or S2TT
print(text) # Show text output 

s2st_model = torch.jit.load("unity_on_device.ptl")
with torch.no_grad():
    text, units, waveform = s2st_model(audio_input, tgt_lang=TGT_LANG) # S2ST model also returns waveform
print(text)
torchaudio.save(f"{OUTPUT_FOLDER}/result.wav", waveform.unsqueeze(0), sample_rate=16000) # Save output waveform to local file
```


Also running the exported model doesn't need python runtime. For example, you could load this model in C++ following [this tutorial](https://pytorch.org/tutorials/advanced/cpp_export.html), or building your own on-device applications similar to [this example](https://github.com/pytorch/ios-demo-app/tree/master/SpeechRecognition)


## Metrics
### S2TT BLEU / S2ST ASR-BLEU on FLEURS
For ASR-BLEU, we follow the same protocol as SeamlessM4T Large/Medium models: We used Whisper-large-v2 for Eng-X and Whisper-medium for X-Eng when evaluating ASR BLEU.
| Direction  | 1st-pass BLEU (S2TT) | 2nd-pass ASR-BLEU (S2ST)
|---------|----------------------|----------------------|
| eng-hin|10.43|15.06|
| eng-por|21.54|17.35|
| eng-rus|7.88|5.11|
| eng-spa|12.78|11.75|
| hin-eng|12.92|10.50|
| por-eng|22.99|24.81|
| rus-eng|18.24|18.24|
| spa-eng|14.37|14.85|

### ASR WER on FLEURS
| LANG  | WER |
|---------|----------------------|
| eng|27.3|
| hin|41.5|
| por|25.2|
| rus|33.0|
| spa|18.0|
