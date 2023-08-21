## Evaluation protocols for various SeamlessM4T tasks
Refer to the [inference tutorial](../../scripts/m4t/predict/README.md) for detailed guidance on how to run inference using SeamlessM4T models. In this tutorial, the evaluation protocol used for all tasks supported by SeamlessM4T is briefly described.

### S2TT
[Sacrebleu library](https://github.com/mjpost/sacrebleu) is used to compute the BLEU scores. To be consistent with Whisper, a character-level (*char*) tokenizer for Mandarin Chinese (cmn), Japanese (jpn), Thai (tha), Lao (lao), and Burmese (mya) is used. The default *13a* tokenizer is used for other languages. Raw (unnormalized) references and predictions are used for computing the scores.

```python
import sacrebleu

bleu_metric = sacrebleu.BLEU(tokenize=<TOKENIZER>)
bleu_score = bleu_metric.corpus_score(<PREDICTIONS>, [<REFERENCES>])
```

### S2ST and T2ST
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
