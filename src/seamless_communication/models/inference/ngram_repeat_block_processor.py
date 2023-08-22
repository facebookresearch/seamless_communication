from fairseq2.generation.logits_processor import LogitsProcessor as LogitsProcessor
from typing import List
from torch import Tensor
import torch


class NGramRepeatBlockProcessor(LogitsProcessor):
    def __init__(self, no_repeat_ngram_size: int) -> None:
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, seqs: Tensor, lprobs: Tensor) -> None:
        """Remove repeating n-gram tokens."""
        batch_size, beam_size, vocab_size = lprobs.size()
        step_nr = seqs.size(2) - 1
        # (N, B, S) -> (N * B, S)
        seqs = seqs.view(-1, seqs.size(2))
        # (N, B, V) -> (N * B, V)
        lprobs = lprobs.view(-1, vocab_size)
        self._no_repeat_ngram(seqs, lprobs, batch_size, beam_size, step_nr)

    def _no_repeat_ngram(
        self,
        seqs: Tensor,
        lprobs: Tensor,
        batch_size: int,
        beam_size: int,
        step_nr: int,
    ) -> Tensor:
        """For each hypothesis generate a list of previous ngrams
            and set associated lprobs to -inf

        :param seqs: The generated sequences of tokens for the first
            `step_nr` steps of decoding (N * B, step_nr + 1)
        :param lprobs: The next-step log probability reshaped to (N * B, V)
        :param batch_size: The batch size.
        :param beam_size: The beam size.
        :param step_nr: Step number for decoding.

        :returns:
            modified lprobs tensor with banned tokens set to -inf
        """
        banned_tokens = [[] for _ in range(batch_size * beam_size)]

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
            lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -torch.inf
        return lprobs
