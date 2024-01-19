# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.
#
# This script contains the builder and loader for the MT models. It has some
# overlaps with fairseq2.models.nllb, except for a few subtle changes
# in the tokenizer, patches of layers, etc.

from pathlib import Path
from typing import Any, Mapping, Optional, Literal
import torch
from torch.nn.parameter import Parameter

from fairseq2.assets import InProcAssetMetadataProvider, asset_store, download_manager
from fairseq2.generation.beam_search import BeamSearchSeq2SeqGenerator
from fairseq2.nn.embedding import StandardEmbedding
from fairseq2.models.nllb.builder import NllbBuilder, NllbConfig
from fairseq2.models.nllb.loader import load_nllb_config
from fairseq2.nn.projection import TiedProjection
from fairseq2.models.transformer.model import TransformerModel
from fairseq2.models.utils import ModelLoader
from fairseq2.typing import Device, DataType
from fairseq2.models.utils.checkpoint import convert_fairseq_checkpoint

import sentencepiece as spm


class MTBuilder(NllbBuilder):
    def build_embedding(self) -> StandardEmbedding:
        return StandardEmbedding(
            num_embeddings=self.config.vocab_info.size,
            embedding_dim=self.config.model_dim,
            pad_idx=self.config.vocab_info.pad_idx,
            init_fn=lambda x: x,
            device=self.device,
            dtype=self.dtype,
        ).requires_grad_(False)

    def build_model(self) -> TransformerModel:
        """Build a model."""
        encoder_embed = self.build_embedding()
        decoder_embed = self.build_embedding()

        encoder_frontend = self.build_frontend(encoder_embed)
        decoder_frontend = self.build_frontend(decoder_embed)

        encoder = self.build_encoder()
        decoder = self.build_decoder()

        # Unlike NLLB, in MT we de-couple
        new_weight = Parameter(torch.zeros_like(
            encoder_embed.weight, requires_grad=False)
        )
        final_proj = TiedProjection(new_weight, bias=None)

        return TransformerModel(
            encoder_frontend,
            encoder,
            decoder_frontend,
            decoder,
            final_proj,
            self.config.vocab_info,
        )


def create_mt_model(
    config: NllbConfig,
    *,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> TransformerModel:
    return MTBuilder(config, device=device, dtype=dtype).build_model()


def convert_mt_checkpoint(
    ckpt: Mapping[str, Any], config: NllbConfig,
) -> Mapping[str, Any]:
    global_key_map = {
        # fmt: off
        r"^encoder\.embed_tokens\.":                              r"encoder_frontend.embed.",
        r"^decoder\.embed_tokens\.":                              r"decoder_frontend.embed.",
        r"^encoder\.embed_positions.weights":                     r"encoder_frontend.pos_encoder.freqs",
        r"^decoder\.embed_positions.weights":                     r"decoder_frontend.pos_encoder.freqs",
        r"^encoder\.layernorm_embedding\.":                       r"encoder_frontend.layer_norm.",
        r"^decoder\.layernorm_embedding\.":                       r"decoder_frontend.layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"decoder.layers.\1.self_attn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.":     r"encoder.layers.\1.self_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.out_proj\.":  r"decoder.layers.\1.encoder_decoder_attn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn\.":            r"decoder.layers.\1.encoder_decoder_attn.",
        r"^decoder\.layers\.([0-9]+)\.encoder_attn_layer_norm\.": r"decoder.layers.\1.encoder_decoder_attn_layer_norm.",
        r"^encoder\.layers\.([0-9]+)\.fc1\.":                     r"encoder.layers.\1.ffn.inner_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc1\.":                     r"decoder.layers.\1.ffn.inner_proj.",
        r"^encoder\.layers\.([0-9]+)\.fc2\.":                     r"encoder.layers.\1.ffn.output_proj.",
        r"^decoder\.layers\.([0-9]+)\.fc2\.":                     r"decoder.layers.\1.ffn.output_proj.",
        r"^encoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"encoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.layers\.([0-9]+)\.final_layer_norm\.":        r"decoder.layers.\1.ffn_layer_norm.",
        r"^decoder\.output_projection\.":                         r"final_proj.",
        # fmt: on
    }
    return convert_fairseq_checkpoint(ckpt, global_key_map)


def load_vocab(model_dir: str, mode: Literal["src", "tgt"]):
    vocab_file = f"{model_dir}/{mode}.spm"
    spmp = spm.SentencePieceProcessor(vocab_file)

    return [
        (spmp.id_to_piece(id).replace("▁", " "), spmp.get_score(id))
        for id in range(spmp.get_piece_size())
    ], spmp


def load_mt_model(model_dir: str):
    """
    Load MT model and the vocabulary processors (spm) for source and target languages
    Args:
        model_dir: Directory of the model. It must contain files averaged_checkpoint.pt, src.spm and tgt.spm
    """

    # Create a fairseq2 model card on the fly. This must ensure that we do not have any other fairseq2
    # environment resolvers and always return
    model_dir = Path(model_dir)
    model_card_info = [
        {
            "name": "mt_model",
            "model_type": "nllb",  # Re-use the same encoder-decoder arch of NLLB
            "model_arch": "dense_600m",  # Dummy value to pass fairseq2 asset's valdilation logic
            "checkpoint": "file://" + str(model_dir / "averaged_checkpoint.pt"),
            "model_config": {
                "model_dim": 512,
                "num_encoder_layers": 4,
                "num_decoder_layers": 2,
                "ffn_inner_dim": 2048,
                "vocab_info": {
                    "size": 10000,
                    "unk_idx": 3,
                    "bos_idx": 0,
                    "eos_idx": 2,
                    "pad_idx": 1,
                }
            }
        }
    ]
    asset_store.metadata_providers.append(
        InProcAssetMetadataProvider(model_card_info)
    )
    mt_card = asset_store.retrieve_card("mt_model")

    return ModelLoader[TransformerModel, NllbConfig](
        asset_store,
        download_manager,
        load_nllb_config,
        create_mt_model,
        convert_mt_checkpoint,
        restrict_checkpoints=False,
    )(mt_card)


def test_mt(
    model: TransformerModel,
    src_spm: spm.SentencePieceProcessor,
    tgt_spm: spm.SentencePieceProcessor,
):
    from fairseq2.nn.padding import pad_seqs

    # Tokens of "This is an example"
    src_tokens = torch.LongTensor([688, 153, 62, 4581, 2])
    src_seqs, src_padding_mask = pad_seqs(src_tokens, src_spm.pad_id())

    # Force the developer begins with the EOS <s> token
    prompt_tokens = torch.LongTensor([[2]])
    generator = BeamSearchSeq2SeqGenerator(model)
    output = generator(src_seqs, src_padding_mask, prompt_tokens, None)

    print(output.hypotheses[0][0].seq)
    tgt_tokens = output.hypotheses[0][0].seq.tolist()
    out_text = tgt_spm.decode(tgt_tokens)

    # assert out_text == "Este es un ejemplo"
    print(out_text)
