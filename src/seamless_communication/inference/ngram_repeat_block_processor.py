# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch
from fairseq2.generation import StepProcessor
from torch import Tensor


class NGramRepeatBlockProcessor(StepProcessor):
    def __init__(self, no_repeat_ngram_size: int) -> None:
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, seqs: Tensor, probs: Tensor, lprob: bool = False) -> None:
        """Remove repeating n-gram tokens."""
        batch_size, beam_size, vocab_size = probs.size()
        step_nr = seqs.size(2) - 1
        # (N, B, S) -> (N * B, S)
        seqs = seqs.view(-1, seqs.size(2))
        # (N, B, V) -> (N * B, V)
        probs = probs.view(-1, vocab_size)
        self._no_repeat_ngram(seqs, probs, lprob, batch_size, beam_size, step_nr)

    def _no_repeat_ngram(
        self,
        seqs: Tensor,
        probs: Tensor,
        lprob: bool,
        batch_size: int,
        beam_size: int,
        step_nr: int,
    ) -> None:
        """For each hypothesis generate a list of previous ngrams
            and set associated lprobs to -inf

        :param seqs: The generated sequences of tokens for the first
            `step_nr` steps of decoding (N * B, step_nr + 1)
        :param probs: The next-step probability reshaped to (N * B, V)
        :param lprob: If ``True``, ``probs`` is log probabilities.
        :param batch_size: The batch size.
        :param beam_size: The beam size.
        :param step_nr: Step number for decoding.
        """
        banned_tokens: List[List[int]] = [[] for _ in range(batch_size * beam_size)]

        if step_nr + 2 - self.no_repeat_ngram_size >= 0:
            cpu_tokens: List[List[int]] = seqs.cpu().tolist()
            check_start_pos = step_nr + 2 - self.no_repeat_ngram_size
            for bbsz_idx in range(batch_size * beam_size):
                ngram_to_check = cpu_tokens[bbsz_idx][
                    -(self.no_repeat_ngram_size - 1) :
                ]
                for i in range(check_start_pos):
                    if (
                        ngram_to_check
                        == cpu_tokens[bbsz_idx][i : i + self.no_repeat_ngram_size - 1]
                    ):
                        banned_tokens[bbsz_idx].append(
                            cpu_tokens[bbsz_idx][i + self.no_repeat_ngram_size - 1]
                        )
        for bbsz_idx in range(batch_size * beam_size):
            probs[bbsz_idx, banned_tokens[bbsz_idx]] = -torch.inf if lprob else 0
