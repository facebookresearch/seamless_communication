# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Transcriber time stamp extraction fine tuning module
"""

from math import isclose
from typing import Union, List, Tuple, Dict

from seamless_communication.models.inference.transcriber import TranscriptionToken

PUNCTUATION_MARKS = "¿?¡!.,:;"
MAX_TIME_DELTA = 5


class FTWord:
    """Transcription Token class with '==' operator functionality"""

    def __init__(
        self,
        text: str = "",
        prob: float = -1,
        ini_time: float = -1,
        end_time: float = -1,
        token: TranscriptionToken = None,
    ):
        if token is not None:
            text = token.text
            prob = token.prob
            ini_time = token.time_s

        self.text = self._clean_text(text)
        self.prob = prob
        self.ini_time = ini_time
        self.end_time = end_time

    @classmethod
    def _clean_text(cls, text):
        """Lowercase, remove punctuation marks, and leading and trailing spaces"""
        text = text.lower()
        text = text.replace("</s>", "")
        for char in PUNCTUATION_MARKS:
            text = text.replace(char, "")
        text = text.strip()
        return text

    def __eq__(self, other: Union[TranscriptionToken, str]):
        if isinstance(other, TranscriptionToken) or isinstance(other, FTWord):
            return self.text == other.text
        elif isinstance(other, str):
            return self.text == other
        else:
            raise NotImplementedError


class FTError:
    time_delta_r: float = 0
    time_delta_l: float = 0
    word_count_r: int = 0
    word_count_l: int = 0
    missing_word_count: int = -1

    def add_time_delta(self, time_delta: float) -> None:
        if time_delta < 0:
            self.time_delta_l += max(time_delta, -MAX_TIME_DELTA)
            self.word_count_l += 1
        else:
            self.time_delta_r += min(time_delta, MAX_TIME_DELTA)
            self.word_count_r += 1

    def get_time_delta_abs(self):
        """Get average absolute time delta including MAX_TIME_DELTA penalty for missed words"""
        return (
            abs(self.time_delta_l)
            + self.time_delta_r
            + self.missing_word_count * MAX_TIME_DELTA
        ) / ((self.word_count_l + self.word_count_r + self.missing_word_count) or 1)

    def get_time_delta_abs_skip_missing(self):
        """Get average absolute time delta without penalizing missed words"""
        return (abs(self.time_delta_l) + self.time_delta_r) / (
            (self.word_count_l + self.word_count_r) or 1
        )

    def get_time_delta_l(self):
        return abs(self.time_delta_l) / (self.word_count_l or 1)

    def get_time_delta_r(self):
        return self.time_delta_r / (self.word_count_r or 1)

    def __repr__(self):
        return (
            f"time delta (abs): {self.get_time_delta_abs()}\n"
            f"time delta (l):   {self.get_time_delta_l()}\n"
            f"time delta (r):   {self.get_time_delta_r()}\n"
            f"missed words:     {self.missing_word_count}"
        )


class FTTranscription:
    words: List[FTWord]

    def __init__(self, words: List[FTWord], lang: str = "", path: str = "") -> None:
        self.words = self.separate_words(words)
        self.lang = lang
        self.path = path

    @staticmethod
    def separate_words(words: List[FTWord]) -> List[FTWord]:
        """Separate FTWords from a list into FTWords containing a single word"""
        separated_words = []
        for word in words:
            texts = word.text.split()
            if len(texts) > 1:
                separated_words.extend(
                    [
                        FTWord(
                            text=text,
                            prob=word.prob,
                            ini_time=word.ini_time,
                            end_time=word.end_time,
                        )
                        for text in texts
                    ]
                )
            else:
                separated_words.append(word)
        return separated_words

    def get_lcs_matrix(self, words: List[FTWord]) -> List[List[int]]:
        """Calculate longest common subsequence matrix"""

        # Initialize matrix
        matrix = [[0] * (len(words) + 1)] * (len(self.words) + 1)

        # Fill matrix
        for i, i_word in enumerate(self.words):
            for j, j_word in enumerate(words):
                if i_word == j_word:
                    matrix[i + 1][j + 1] = matrix[i][j] + 1
                else:
                    matrix[i + 1][j + 1] = max(matrix[i + 1][j], matrix[i][j + 1])

        return matrix

    def get_lcs(
        self,
        matrix: List[List[int]],
        words: List[FTWord],
        i: Union[int, None] = None,
        j: Union[int, None] = None,
    ) -> List[Tuple[FTWord, FTWord]]:
        """Get longest common subsequence of words"""

        # Initialize pointers
        if i is None:
            i = len(self.words)
        if j is None:
            j = len(words)

        # Base case
        if i == 0 or j == 0:
            return []

        # Read matrix
        if self.words[i - 1] == words[j - 1]:
            return [
                *self.get_lcs(matrix, words, i - 1, j - 1),
                (self.words[i - 1], words[j - 1]),
            ]
        elif matrix[i][j - 1] > matrix[i - 1][j]:
            return self.get_lcs(matrix, words, i, j - 1)
        else:
            print(f"No match (j): {self.words[i - 1].text}, {words[j - 1].text}")
            return self.get_lcs(matrix, words, i - 1, j)

    def compare(self, words: List[FTWord]) -> FTError:
        """Calculate time delta and missed words between lists of words"""
        words = self.separate_words(words)
        lcs_matrix = self.get_lcs_matrix(words)
        lcs = self.get_lcs(lcs_matrix, words)
        err = FTError()
        for w1, w2 in lcs:
            t = w1.ini_time - w2.ini_time
            if not isclose(t, 0):
                err.add_time_delta(t)
        err.missing_word_count = len(self.words) - len(lcs)
        return err


class FineTuneTranscriber:
    def __init__(self, transcriptions: List[Dict]) -> None:
        """
        Input: list of transcriptions
        [
            {
                "audio_path": "/path/to/audio.file",
                "text": "transcription",
                "language": "eng",
                "words": [
                    {
                        "word": "transcription",
                        "start": 0.1,
                        "end": 0.3,
                        "probability": 0.9
                    },
                ]
            },
        ]
        """
        self.transcriptions = list()
        for transcription in transcriptions:
            self.transcriptions.append(
                FTTranscription(
                    [
                        FTWord(
                            text=word["word"],
                            prob=word["probability"],
                            ini_time=word["start"],
                            end_time=word["end"],
                        )
                        for word in transcription["words"]
                    ],
                    lang=transcription["lang"],
                    path=transcription["audio_path"],
                )
            )
