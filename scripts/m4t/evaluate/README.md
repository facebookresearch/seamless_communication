# Evaluating SeamlessM4T models
Refer to the [inference tutorial](../predict/README.md) for the supported tasks to run inference with SeamlessM4T models.

## Quick start:
Evaluation can be run with the CLI, from the root directory of the repository.

The model can be specified with `--model_name`: `seamlessM4T_v2_large` or `seamlessM4T_large` or `seamlessM4T_medium`

```bash
m4t_evaluate <path_to_data_tsv_file> <task_name> <tgt_lang> --output_path <path_to_save_evaluation_output> --ref_field <ref_field_name> --audio_root_dir <path_to_audio_root_directory>
```

### S2TT
If provided a test_fleurs/dev_fleurs data tsv file, we parse through every example in the file, run model inference and save the first pass text generations and the computed first pass (S2TT) BLEU.

### S2ST and T2ST
Additionally from S2TT, we also save the unit generations, run vocoder inference to generate the translated audio waveforms and save the .wav files to a directory.

To measure the quality of the translated speech outputs, the audios are first transcribed using Whisper ASR model and BLEU score is computed on these ASR transcriptions comparing them with the ground truth text references.

Whisper large-v2 is used for non-English target languages and medium.en trained on English-only data is used for English due to its superior performance.

```python
import whisper

model = whisper.load_model('medium.en')
model = whisper.load_model('large-v2')
```
To reproduce the whisper transcriptions and thereby the ASR-BLEU scores, greedy decoding is used with a preset temperature value of 0. Target language information is also passed to the whisper model.

```python
prediction = model.transcribe(<AUDIO_PATH>, language=<LANGUAGE>, temperature=0, beam_size=1)["text"]
```

Whisper-normalizer is run on the ground truth <REFERENCES> and the model generated <PREDICTIONS>. ASR-BLEU scores are computed using sacrebleu following the same tokenization as described for S2TT.

```python
from whisper_normalizer.basic import BasicTextNormalizer

normalizer = EnglishTextNormalizer() ## To be used for English
normalizer = BasicTextNormalizer()  ## For non-English directions
```

### T2TT
Similar to S2TT, raw (unnormalized) references and predictions are used to compute the chrF++ scores for text-to-text translation.

```python
import sacrebleu

chrf_metric = sacrebleu.CHRF(word_order=2)
chrf_score = chrf_metric.corpus_score(<REFERENCES>,<PREDICTIONS>)
```

### ASR
Similar to Whisper, character-level error rate (CER) metric is used for Mandarin Chinese (cmn), Japanese (jpn), Thai (tha), Lao (lao), and Burmese (mya) languages. Word-level error rate (WER) metric is used for the remaining languages. Whisper-normalizer is applied on the ground truth <REFERENCES> and the model generated <PREDICTIONS>. [JiWER library](https://github.com/jitsi/jiwer) is used to compute these CER and WER scores.

```python
import jiwer

wer = WER(<REFERENCES>,<PREDICTIONS>) ## WER
cer = CER(<REFERENCES>,<PREDICTIONS>) ## CER

```
