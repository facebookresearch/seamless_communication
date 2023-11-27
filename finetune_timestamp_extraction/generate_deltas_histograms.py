"""
Generate Deltas Histograms
Input:
    - file(s) in the format of `finetune_timestamp_extraction` output
        ```json
            {
                str(lang) : [
                    [
                        str(transcription_whisper),
                        str(transcription_seamless),
                        int(time_delta)
                    ]
                ],
                "average": { str(lang): int(time_delta) }
                "metadata": {
                    "use_dtw": bool()
                    "median_filter_width": int()
                }
            }
        ```
Output:
    - png file with histogram plots
"""

from datetime import datetime
import json
import matplotlib.pyplot as plt
from sys import argv

deltas = {}

for file in argv[1:]:
    with open(file, mode="r", encoding="utf-16") as file:
        blob = file.read()
        data = json.loads(blob)
        for key, transcriptions in data.items():
            if key in ("average", "metadata"):
                continue
            deltas[key] = [t[2] for t in transcriptions]

    fig, axs = plt.subplots(2, 3, tight_layout=True, sharex=True, sharey=True)

    fig.suptitle(str(data["metadata"]))

    for idx, key in enumerate(deltas.keys()):
        axs[idx % 2, idx // 2].hist(deltas[key], bins=10)
        axs[idx % 2, idx // 2].set_title(key)
        axs[idx % 2, idx // 2].set_xlabel("delta seconds")
        axs[idx % 2, idx // 2].set_ylabel("occurences")

    plt.savefig(f"hist_{int(datetime.now().timestamp())}.png")
