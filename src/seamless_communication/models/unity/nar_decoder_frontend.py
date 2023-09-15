# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, final

from torch import Tensor
from torch.nn import Dropout, Module, Parameter

from fairseq2.data import VocabularyInfo
from fairseq2.data.text import TextTokenizer
from fairseq2.models.nllb.tokenizer import NllbTokenizer
from fairseq2.nn.embedding import Embedding
from fairseq2.nn.normalization import LayerNorm
from fairseq2.nn.position_encoder import PositionEncoder
from fairseq2.nn.transformer import create_default_layer_norm
from fairseq2.nn.utils.mask import to_padding_mask
from fairseq2.typing import DataType, Device, finaloverride


from seamless_communication.models.unity.length_regulator import (
    HardUpsampling,
    VarianceAdaptor,
)
from seamless_communication.models.unity.char_tokenizer import CharTokenizer

import math
import torch


SPACE = "▁"


class TagManager:
    def __init__(self, text_tokenizer: TextTokenizer):
        self.vocab_info: VocabularyInfo = text_tokenizer.vocab_info
        token_encoder = text_tokenizer.create_encoder(mode="target")
        self.prefix_len: int = (
            token_encoder.prefix_indices.shape[0]
            if token_encoder.prefix_indices is not None
            else 0
        )
        self.suffix_len: int = (
            token_encoder.suffix_indices.shape[0]
            if token_encoder.suffix_indices is not None
            else 0
        )

    def preprocess_text_seqs(self, text_seqs: Tensor) -> Tensor:
        """Remove the prefix, suffix tokens."""
        seq_len = text_seqs.shape[1]
        text_seqs = text_seqs[:, self.prefix_len : seq_len - self.suffix_len]
        assert self.vocab_info.pad_idx is not None
        text_seqs.masked_fill_(
            text_seqs == self.vocab_info.eos_idx, self.vocab_info.pad_idx
        )
        return text_seqs

    def postprocess_dur_or_len(self, dur_or_len: Tensor) -> Tensor:
        """Add back 0s in place of the prefix, suffix tokens."""
        N = dur_or_len.shape[0]

        prefix = dur_or_len.new_zeros((N, self.prefix_len))
        suffix = dur_or_len.new_zeros((N, self.suffix_len))

        dur_or_len = torch.cat([prefix, dur_or_len, suffix], dim=1)
        return dur_or_len


@final
class NARDecoderFrontend(Module):
    """Represents a Non-autoregressive decoder front-end."""

    char_pos_encoder: PositionEncoder
    pos_emb_alpha_char: Parameter
    unit_pos_encoder: PositionEncoder
    pos_emb_alpha: Parameter
    scale: float
    char_length_regulator: HardUpsampling
    variance_adaptor: VarianceAdaptor
    layer_norm: Optional[LayerNorm]
    dropout: Optional[Dropout]

    def __init__(
        self,
        embed: Embedding,
        embed_char: Embedding,
        text_tokenizer: NllbTokenizer,
        char_tokenizer: CharTokenizer,
        unit_pos_encoder: PositionEncoder,
        char_pos_encoder: PositionEncoder,
        variance_adaptor: VarianceAdaptor,
        no_scale: bool = False,
        layer_norm: bool = False,
        dropout_p: float = 0.1,
        device: Optional[Device] = None,
        dtype: Optional[DataType] = None,
    ):
        self.model_dim = embed.embedding_dim

        super().__init__()

        self.embed = embed
        self.embed_char = embed_char
        self.text_tokenizer = text_tokenizer
        self.char_tokenizer = char_tokenizer
        self.tag_manager = TagManager(text_tokenizer)

        # TODO: Remove this hack once we get the correct char SPM .model file.
        from fairseq.data import Dictionary

        dict_file = "/checkpoint/krs/unity2/unity2_data/spm_char_lang38_tc.txt"
        self.char_dict = Dictionary.load(dict_file)

        self.unk_idx = self.text_tokenizer.vocab_info.unk_idx
        self.pad_idx = self.text_tokenizer.vocab_info.pad_idx

        # TODO: Implement AlignmentEncoder for training.

        if unit_pos_encoder.encoding_dim != self.model_dim:
            raise ValueError(
                f"`encoding_dim` of `unit_pos_encoder` and `embedding_dim` of `embed` must be equal, but are {unit_pos_encoder.encoding_dim} and {self.model_dim} instead."
            )

        if char_pos_encoder.encoding_dim != self.model_dim:
            raise ValueError(
                f"`encoding_dim` of `char_pos_encoder` and `embedding_dim` of `embed` must be equal, but are {char_pos_encoder.encoding_dim} and {self.model_dim} instead."
            )

        self.unit_pos_encoder = unit_pos_encoder

        self.pos_emb_alpha = Parameter(torch.ones(1, device=device, dtype=dtype))
        self.char_pos_encoder = char_pos_encoder

        self.pos_emb_alpha_char = Parameter(torch.ones(1, device=device, dtype=dtype))
        self.scale = 1.0 if no_scale else math.sqrt(self.model_dim)

        self.char_length_regulator = HardUpsampling()

        self.variance_adaptor = variance_adaptor

        if layer_norm:
            self.layer_norm = create_default_layer_norm(
                self.model_dim, device=device, dtype=dtype
            )
        else:
            self.register_module("layer_norm", None)

        if dropout_p > 0.0:
            self.dropout = Dropout(dropout_p)
        else:
            self.register_module("dropout", None)

    def indices_to_subwords(self, text_seqs: Tensor) -> List[List[str]]:
        N, seq_len = text_seqs.shape
        subwords_batch = []
        for b in range(N):
            subwords = []
            for i in range(seq_len):
                subword = self.text_tokenizer.model.index_to_token(int(text_seqs[b, i]))
                subwords.append(str(subword))
            subwords_batch.append(subwords)
        return subwords_batch

    def text_to_char_seqs(self, text_seqs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        text_seqs = self.tag_manager.preprocess_text_seqs(text_seqs)

        subwords_batch = self.indices_to_subwords(text_seqs)

        char_lens = self.count_character_length_in_subword(text_seqs, subwords_batch)

        char_lens = self.tag_manager.postprocess_dur_or_len(char_lens)

        char_seqs, char_seq_lens = self.get_char_seqs(
            text_seqs, subwords_batch, char_lens
        )

        return char_seqs, char_seq_lens, char_lens

    def count_character_length_in_subword(
        self,
        text_seqs: Tensor,
        subwords_batch: List[List[str]],
        merge_space_with_prev_subword: bool = False,
    ) -> Tensor:
        N, _ = text_seqs.shape

        char_lens = text_seqs.new_zeros(text_seqs.size())

        assert self.pad_idx is not None
        subword_lens = text_seqs.ne(self.pad_idx).sum(1)

        for b in range(N):
            # We slice out the tensor till the padding index.
            subword_indices = text_seqs[b, : subword_lens[b]]
            subwords = subwords_batch[b][: subword_lens[b]]

            assert subword_indices.shape[0] == len(subwords)

            is_next_start_with_space = [
                len(subwords[i + 1]) > 1 and subwords[i + 1][0] == SPACE
                if i < len(subwords) - 1
                else False
                for i in range(len(subwords))
            ]
            is_punc = [
                len(subwords[i]) == 1
                and not subwords[i].isalpha()
                and not subwords[i].isnumeric()
                and subwords[i] != SPACE
                for i in range(len(subwords))
            ]
            for i, (subword_idx, subword) in enumerate(zip(subword_indices, subwords)):
                if subword_idx == self.pad_idx:
                    break

                if subword_idx == self.unk_idx:
                    # We set char_len to 1 for an unk token.
                    char_len = 1

                    if merge_space_with_prev_subword and is_next_start_with_space[i]:
                        char_len += 1
                else:
                    # By default, spaces are merged with the next subword.
                    # char_len includes the space.
                    char_len = len(subword)

                    if merge_space_with_prev_subword:
                        # Add the space for the next subword.
                        if is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the current subword.
                        if i > 0 and is_next_start_with_space[i - 1]:
                            char_len -= 1
                    else:
                        # Merge space with punctuation mark by default.
                        if is_punc[i] and is_next_start_with_space[i]:
                            char_len += 1
                        # Subtract the space for the subword succeeding the punctuation mark.
                        elif (
                            i > 0 and is_punc[i - 1] and is_next_start_with_space[i - 1]
                        ):
                            char_len -= 1

                char_lens[b, i] = char_len

        return char_lens

    def get_char_seqs(
        self, text_seqs: Tensor, subwords_batch: List[List[str]], char_lens: Tensor
    ) -> Tuple[Tensor, Tensor]:
        N = text_seqs.shape[0]
        max_len = int(char_lens.sum(1).max().item())

        assert self.pad_idx is not None
        char_seqs = text_seqs.new_zeros((N, max_len)).fill_(self.pad_idx)
        char_seq_lens = char_seqs.new_zeros(N)

        assert self.pad_idx is not None
        subword_lens = text_seqs.ne(self.pad_idx).sum(1)

        for b in range(N):
            total = 0
            subword_indices = text_seqs[b, : subword_lens[b]]
            subwords = subwords_batch[b][: subword_lens[b]]
            for subword_idx, subword in zip(subword_indices, subwords):
                if subword_idx == self.unk_idx:
                    char_ids = [self.unk_idx]
                else:
                    # Get char token indices corresponding to the subwords.
                    # char_ids = [
                    #     self.char_tokenizer.model.token_to_index(ch)
                    #     for ch in list(subword)
                    # ]
                    # TODO: Remove this hack once we get the correct char SPM .model file.
                    char_ids = [self.char_dict.index(ch) for ch in list(subword)]
                char_seq_len = len(char_ids)
                char_seqs[b, total : total + char_seq_len] = torch.tensor(char_ids).to(
                    char_seqs
                )
                total += char_seq_len
            char_seq_lens[b] = total
        return char_seqs, char_seq_lens

    def character_level_upsampling(
        self,
        seqs: Tensor,
        padding_mask: Optional[Tensor],
        char_seqs: Tensor,
        char_lens: Tensor,
    ) -> Tensor:
        seqs, _ = self.char_length_regulator(seqs, char_lens)

        pos_embeds = self.pos_emb_alpha_char * (
            self.char_pos_encoder(seqs, padding_mask) - seqs
        )

        char_embeds = self.embed_char(char_seqs)
        print(f"char_embeds: {char_embeds.sum()}, {char_embeds.shape}")

        if self.scale != 1.0:
            char_embeds *= self.scale

        pos_embeds += char_embeds

        seqs += pos_embeds

        return seqs

    def forward_unit_pos_embedding(
        self, seqs: Tensor, padding_mask: Optional[Tensor]
    ) -> Tensor:
        pos_embeds = self.pos_emb_alpha * (
            self.unit_pos_encoder(seqs, padding_mask) - seqs
        )

        seqs += pos_embeds

        if self.dropout is not None:
            seqs = self.dropout(seqs)

        return seqs

    @finaloverride
    def forward(
        self,
        target_seqs: Optional[Tensor],
        target_seq_lens: Optional[Tensor],
        encoder_output: Tensor,
        encoder_padding_mask: Optional[Tensor],
        text_seqs: Optional[Tensor],
    ) -> Tuple[Tensor, Optional[Tensor]]:
        assert text_seqs is not None

        # text_seqs: (N, S_text)
        char_seqs, char_seq_lens, char_lens = self.text_to_char_seqs(text_seqs)

        # char_seqs: (N, S_char)
        encoder_padding_mask = to_padding_mask(char_seqs, char_seq_lens)

        # (N, S_text, M) -> (N, S_char, M)
        seqs = self.character_level_upsampling(
            encoder_output, encoder_padding_mask, char_seqs, char_lens
        )

        # (N, S_char, M) -> (N, S_unit, M)
        seqs, seq_lens = self.variance_adaptor(
            seqs,
            encoder_padding_mask,
            min_duration=1,
        )

        decoder_padding_mask = to_padding_mask(seqs, seq_lens)

        seqs = self.forward_unit_pos_embedding(seqs, decoder_padding_mask)

        return seqs, decoder_padding_mask
