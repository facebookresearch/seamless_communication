"""
Fine Tune Timestamp Extraction
Load transcriptions pre-made with Whisper, run audio through Seamless and compare timestamps.
It is expected that the files `transcriptions/whisper_{lang}.json` contain data in the format:
```json
[
    {
        "audio_path": "/path/to/file.wav",
        "text": "transcribed text",
        "words": [
            {
                "word": "transcribed",
                "start": 0.1,
                "end": 0.25,
                "probability": 0.8
            },
            ...
        ]
    },
    ...
]
```
"""

import json

from FineTuneTranscriber import FineTuneTranscriber
from seamless_communication.models.inference import Transcriber

WHISPER_TO_SEAMLESS = {
    "de": "deu",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "ru": "rus",
}
PATH = "transcriptions"
MODEL = Transcriber("seamlessM4T_medium")

# Load text pre-transcribed with Whisper
transcriptions = list()
for w_lang, s_lang in WHISPER_TO_SEAMLESS.items():
    with open(f"{PATH}/whisper_{w_lang}.json", "r", encoding="utf-16") as file:
        current_transcriptions = json.loads(file.read())
        for transcription in current_transcriptions:
            transcription["lang"] = s_lang
            transcriptions.append(transcription)

ftt = FineTuneTranscriber(MODEL, transcriptions)

results = ftt.compare(use_dtw=True)

with open("results.txt", "w", encoding="utf-16") as file:
    file.write(json.dumps(results, indent=2, ensure_ascii=False))
