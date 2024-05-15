# Running inference

### SeamlessM4T Inference
Here’s an example of using the CLI from the root directory to run inference.

S2ST task:
```bash
m4t_predict <path_to_input_audio> --task s2st --tgt_lang <tgt_lang> --output_path <path_to_save_audio>
```
T2TT task:
```bash
m4t_predict <input_text> --task t2tt --tgt_lang <tgt_lang> --src_lang <src_lang>
```
Please refer to the [inference README](src/seamless_communication/cli/m4t/predict) for detailed instruction on how to run inference and the list of supported languages on the source, target sides for speech, text modalities.

For running S2TT/ASR natively (without Python) using GGML, please refer to [the unity.cpp section](#unitycpp).

# Transcription Utilities: Denoising and Segmentation 

The following information shows how to use denoising and segmentation tools for noisy and long input audios.

## Demucs: Audio Denoising Tool

The 'Demucs' class provides functionality for denoising audio in the transcription pipeline. It supports various configuration options, allowing for fine-tuning denoising performance based on specific requirements. 

Key Features:

- Denoising audio using the Demucs model.
- Configurable parameters for denoising.
- Support for both Tensor input and audio file input.
- Automatic cleanup of temporary files generated during denoising.

### Installation

Manually install demucs: 

pip install git+https://github.com/facebookresearch/demucs#egg=demucs

### Usage

To utilize Demucs for denoising audio, instantiate the Transcriber class and optionally the DenoisingConfig class with desired configuration. 'denoise' parameter is False by default, and needs to be set to True to use denoising.

```
import torch
from seamless_communication.inference import Transcriber
from seamless_communication.denoise.demucs import DenoisingConfig

model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

transcriber = Transcriber (
    model_name,
    device=torch.device("cpu"),
    dtype=torch.float32,
)

denoise_config = DenoisingConfig(float32= True)

txt = transcriber.transcribe(audio="example.wav", src_lang="eng", denoise=True, denoise_config=denoise_config)
```

## Silero VAD Segmenter: Audio Segmentation Tool

The 'SileroVADSegmenter' class offers functionality for segmenting long audio recordings into chunks in the transcription pipeline. This tool segments based on speech timestamps. 

Key Features:

- Segmenting long audio recordings into chunks based on speech presence.
- Automatic segmenting of all audio longer than the chunk size. 
- Configurable parameters for chunk size and pause length.
- Resampling audio to match the model's sample rate.
- Efficient speech probability computation using sliding windows.

### Usage

To utilize Silero VAD for segmenting audio, instantiate the Transcriber class. When using the transcribe method, audio will be segmented automatically if it is longer than chunk_size_sec, which has a default value of 20. Use a smaller value for better quality transcription.
pause_length_sec determines the duration of silence between segments and has a default value of 1 second. This parameter can be customized.

```
import torch
from seamless_communication.inference import Transcriber

model_name = "seamlessM4T_v2_large"
vocoder_name = "vocoder_v2" if model_name == "seamlessM4T_v2_large" else "vocoder_36langs"

transcriber = Transcriber (
    model_name,
    device=torch.device("cpu"),
    dtype=torch.float32,
)

input_audio = "example.wav"

txt = transcriber.transcribe(audio=input_audio, src_lang="eng", chunk_size_sec=10, pause_length_sec=.5)
```