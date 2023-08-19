## Evaluation protocols for various SeamlessM4T tasks
Refer to the [inference tutorial](../../scripts/m4t/predict/README.md) for detailed guidance on how to run inference using SeamlessM4T models. In this tutorial, the evaluation protocol used for all tasks supported by SeamlessM4T is briefly described.

### S2TT
Sacrebleu library is used to compute the BLEU scores. To be consistent with Whisper, a character-level(*char*) tokenizer for Mandarin Chinese (cmn), Japanese (jpn), Thai (tha), Lao (lao), and Burmese (mya) is used, and the default *13a* tokenizer is used for other languages. Raw references and predictions are used for score computation and no normalization is applied.

```python
import sacrebleu

bleu_metric = sacrebleu.BLEU(tokenize=<TOKENIZER>)
bleu_score = bleu_metric.corpus_score(<PREDICTIONS>, [<REFERENCES>])
```

### S2ST and T2ST
To measure the quality of the translated speech outputs, the audios are first transcribed using Whisper ASR model and later run sacrebleu on the ASR transcriptions comparing with the ground truth text references to compute the ASR-BLEU metric.

Whisper large-v2 is used for non-English directions and medium.en trained on English-only data is used for English due to its superior performance.
```python
import whisper

model = whisper.load_model('medium.en')
model = whisper.load_model('large-v2')
```
To reproduce the transcriptions and there-by the ASR-BLEU scores, the language information is passed and the temperature, beam values are preset.

```python
prediction = model.transcribe(<AUDIO_PATH>, language=<LANGUAGE>, temperature=0, beam_size=1)["text"]
```

Whisper-normalizer is run on the ground truth <REFERENCES> and the model generated <PREDICTIONS> and score computation protocol for S2TT is followed to get the S2ST ASR-BLEU score
```python
from whisper_normalizer.basic import BasicTextNormalizer

normalizer = EnglishTextNormalizer() ## To be used for English
normalizer = BasicTextNormalizer()  ## For non-English directions
```

### T2TT
Similar to S2TT, raw references and predictions are used to compute the chrf++ scores for text translation task and no normalization is applied.

```python
import sacrebleu

chrf_metric = sacrebleu.CHRF(word_order=2)
chrf_score = chrf_metric.corpus_score(<REFERENCES>,<PREDICTIONS>)
```
The spBLEU scores are reported for T2TT by using *flores200* tokenizer in sacrebleu.

### ASR
Similar to Whisper, character-level error rates (CER) is used for Mandarin Chinese (cmn), Japanese (jpn), Thai (tha), Lao (lao), and Burmese (mya), languages and word-level error rates (WER) is used for the remaining languages. Whisper-normalizer is applied on the references & predictions and the `jiwer` library is used to compute the CER and WER scores.
```python
import jiwer

wer = WER(<REFERENCES>,<PREDICTIONS>) ## WER
cer = CER(<REFERENCES>,<PREDICTIONS>) ## CER

```
