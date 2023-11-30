# Evaluating SeamlessStreaming and Seamless models
SeamlessStreaming is the streaming only model and Seamless is the expressive streaming model.

## Quick start:

Evaluation can be run with the `streaming_evaluate` CLI.

We use the `seamless_streaming_unity` for loading the speech encoder and T2U models, and `seamless_streaming_monotonic_decoder` for loading the text decoder for streaming evaluation. This is already set as defaults for the `streaming_evaluate` CLI, but can be overridden using the `--unity-model-name` and  `--monotonic-decoder-model-name` args if required.

Note that the numbers in our paper use single precision floating point format (fp32) for evaluation by setting `--dtype fp32`. Also note that the results from running these evaluations might be slightly different from the results reported in our paper (which will be updated soon with the new results).

### S2TT:
Set the task to `s2tt` for evaluating the speech-to-text translation part of the SeamlessStreaming model.

```bash
streaming_evaluate --task s2tt --data-file <path_to_data_tsv_file> --audio-root-dir <path_to_audio_root_directory> --output <path_to_evaluation_output_directory> --tgt-lang <3_letter_lang_code>
```

Note: The `--ref-field` can be used to specify the name of the reference column in the dataset.

### ASR:
Set the task to `asr` for evaluating the automatic speech recognition part of the SeamlessStreaming model. Make sure to pass the source language as the `--tgt-lang` arg.

```bash
streaming_evaluate --task asr --data-file <path_to_data_tsv_file> --audio-root-dir <path_to_audio_root_directory> --output <path_to_evaluation_output_directory> --tgt-lang <3_letter_source_lang_code> 
```

### S2ST:

#### SeamlessStreaming:

Set the task to `s2st` for evaluating the speech-to-speech translation part of the SeamlessStreaming model. 

```bash
streaming_evaluate --task s2st --data-file <path_to_data_tsv_file> --audio-root-dir <path_to_audio_root_directory> --output <path_to_evaluation_output_directory> --tgt-lang <3_letter_lang_code>
```

#### Seamless:
The Seamless model is an unified model for streaming expressive speech-to-speech tranlsation. Use the `--expressive` arg for running evaluation of this unified model.

```bash
streaming_evaluate --task s2st --data-file <path_to_data_tsv_file> --audio-root-dir <path_to_audio_root_directory> --output <path_to_evaluation_output_directory> --tgt-lang <3_letter_lang_code> --expressive
```

The Seamless model uses `vocoder_pretssel` which is a 24KHz version (`vocoder_pretssel`) by default. In the current version of our paper, we use 16KHz version (`vocoder_pretssel_16khz`) for the evaluation , so in order to reproduce those results please add this arg to the above command: `--vocoder-name vocoder_pretssel_16khz`.

`vocoder_pretssel` or `vocoder_pretssel_16khz` checkpoints are gated, please check out [this section](/README.md#seamlessexpressive-models) to acquire these checkpoints. Also, make sure to add `--gated-model-dir <path_to_vocoder_checkpoints_dir>`
