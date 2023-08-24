# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from fairseq2.nn.transformer import TransformerEncoder, TransformerNormOrder
from fairseq2.models.wav2vec2 import (
    Wav2Vec2EncoderConfig,
    Wav2Vec2Config,
    wav2vec2_arch,
    Wav2Vec2Model,
    create_wav2vec2_model,
    Wav2Vec2Frontend,
)
from fairseq2.models.wav2vec2.loader import Wav2Vec2Loader
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.sequence import SequenceBatch


from seamless_communication.assets import asset_store, download_manager


import torch
from typing import Optional

from torch import Tensor
import torch.nn as nn


wav2vec2_archs = ArchitectureRegistry[Wav2Vec2Config]("wav2vec2")
wav2vec2_arch = wav2vec2_archs.marker


def _encoder_xlsr2_1b_v2() -> Wav2Vec2EncoderConfig:
    layer_descs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    return Wav2Vec2EncoderConfig(
        model_dim=1280,
        max_seq_len=4096,
        feature_dim=512,
        use_fbank=False,
        first_pass_dropout_p=0.0,
        layer_norm_features=False,
        feature_extractor_layer_descs=layer_descs,
        feature_extractor_bias=True,
        feature_extractor_layer_norm_convs=True,
        feature_grad_scale=1.0,
        num_fbank_channels=0,
        fbank_stride=0,
        sample_fbank_every_k=0,
        pos_encoder_type="conv",
        pos_encoder_depth=1,
        pos_conv_kernel_size=128,
        num_pos_conv_groups=16,
        use_conformer=False,
        num_encoder_layers=48,
        num_encoder_attn_heads=16,
        ffn_inner_dim=5120,
        dropout_p=0.1,
        attn_dropout_p=0.1,
        layer_drop_p=0.0,
        norm_order=TransformerNormOrder.PRE,
        depthwise_conv_kernel_size=0,
    )


@wav2vec2_arch("xlsr2_1b_v2")
def _xlsr2_1b_v2() -> Wav2Vec2Config:
    encoder_config = _encoder_xlsr2_1b_v2()

    return Wav2Vec2Config(
        encoder_config,
        final_dim=1024,
        final_proj_bias=True,
        temporal_mask_span_len=10,
        max_temporal_mask_prob=0.65,
        spatial_mask_span_len=10,
        max_spatial_mask_prob=0.0,
        quantized_dim=1024,
        num_codebooks=2,
        num_codebook_entries=320,
        codebook_sampling_temperature=(2, 0.1, 0.999995),
        num_distractors=100,
        logit_temp=0.1,
        diversity_loss_weight=0.1,
    )


load_wav2vec2_model = Wav2Vec2Loader(
    asset_store,
    download_manager,
    create_wav2vec2_model,
    wav2vec2_archs,
)


class Wav2Vec2LayerOutputModel(nn.Module):
    encoder_frontend: Wav2Vec2Frontend
    encoder: TransformerEncoder

    def __init__(self, w2v2: Wav2Vec2Model):
        super().__init__()

        self.encoder_frontend = w2v2.encoder_frontend
        self.encoder = w2v2.encoder

    @torch.inference_mode()
    def forward(self, batch: SequenceBatch, out_layer_idx: int):
        """
        :param batch:
            The batch of sequences to process.
        """
        seqs, padding_mask = self.encoder_frontend(batch.seqs, batch.seq_lens)
        w2v2_layer_output = None

        def layer_output_hook(
            layer_idx: int,
            layer_output: Tensor,
            layer_padding_mask: Optional[Tensor],
            num_layers: int,
        ) -> None:
            nonlocal w2v2_layer_output

            if layer_idx == out_layer_idx:
                w2v2_layer_output = layer_output

        _, _ = self.encoder(seqs, padding_mask, layer_output_hook)

        assert w2v2_layer_output is not None
        return w2v2_layer_output
