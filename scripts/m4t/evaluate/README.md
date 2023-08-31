# Evaluation script for SeamlessM4T

## Quick start
Evaluation is run with the CLI, from the root directory of the repository. Currently, the script supports evaluation for S2ST tasks.

### Usage:

```bash
m4t_evaluate <path_to_input_audio> <path_to_reference_text> <tgt_lang> --src_lang <src_lang> --audio_format <audio_format> --output_path <path_to_output_dir> --dataset_name <dataset_name> --save_first_pass <true/false> --model_name <model_name>
```

## Evaluation breakdown
First, the selected SeamlessM4T model translates the audios, outputting the first pass text data and the audio units.

```python
model: UnitYModel = Translator.load_model_for_inference(
    model_name_or_card=model_name,
    #...other fields
)
result = Translator.get_prediction(
    model=model,
    src=src,
    #...other fields
)
```

The default vocoder is used to generate audio from the computed audio units. 

```python
vocoder: Vocoder = Translator.load_model_for_inference(
    load_model_fn=load_vocoder_model,
    model_name_or_card="vocoder_36langs",
    #...other fields
)
wav_out = vocoder(units, tgt_lang, -1, dur_prediction=True)
```

To measure the quality of the translated speech outputs, the vocoder output audios are first transcribed using Whisper ASR model and BLEU score is computed on these ASR transcriptions comparing them with the ground truth text references.

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

Whisper-normalizer is run on the ground truth references and the model generated predictions.

```python
from whisper_normalizer.basic import BasicTextNormalizer

normalizer = EnglishTextNormalizer() ## To be used for English
normalizer = BasicTextNormalizer()  ## For non-English directions
```

ASR-BLEU scores are computed using sacrebleu. To be consistent with Whisper, a character-level (*char*) tokenizer for Mandarin Chinese (cmn), Japanese (jpn), Thai (tha), Lao (lao), and Burmese (mya) is used. The default *13a* tokenizer is used for other languages. BLEU scores are computed for both the first pass text as well as the whisper transcriptions against the ground truth references.

```python
import sacrebleu

bleu_metric = sacrebleu.BLEU(tokenize=<TOKENIZER>)
bleu_score = bleu_metric.corpus_score(<PREDICTIONS>, [<REFERENCES>])
```