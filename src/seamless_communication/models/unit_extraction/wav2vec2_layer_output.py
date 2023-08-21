# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from fairseq2.nn.transformer import TransformerNormOrder
from fairseq2.models.wav2vec2 import (
    Wav2Vec2EncoderConfig,
    Wav2Vec2Config,
    wav2vec2_arch,
    Wav2Vec2Model,
    Wav2Vec2Builder,
    Wav2Vec2EncoderBuilder,
)
from fairseq2.models.wav2vec2.loader import Wav2Vec2Loader
from fairseq2.models.utils.arch_registry import ArchitectureRegistry
from fairseq2.models.utils.model_loader import ModelConfigLoader
from fairseq2.typing import DataType, Device
from fairseq2.models.sequence import SequenceBatch


from seamless_communication.assets import asset_store, download_manager


import torch
from typing import Optional

from torch import Tensor

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
        dropout_p=0.0,
        attn_dropout_p=0.0,
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


class Wav2Vec2LayerOutputModel(Wav2Vec2Model):
    @torch.no_grad()
    def forward(self, batch: SequenceBatch, out_layer_idx: int):
        """
        :param batch:
            The batch of sequences to process.
        """
        print(f"Before run_frontend: {batch.seqs.sum()}")
        seqs, padding_mask, _, _ = self.run_frontend(batch.seqs, batch.seq_lens)
        print(f"After run_frontend: {seqs.sum()}")
        w2v2_layer_output = None

        def layer_output_hook(
            layer_idx: int,
            layer_output: Tensor,
            layer_padding_mask: Optional[Tensor],
            num_layers: int,
        ) -> None:
            nonlocal w2v2_layer_output

            if layer_idx == out_layer_idx:
                print(f"{layer_idx=}")
                w2v2_layer_output = layer_output

        # TODO: Should pad for fp16?
        _, _ = self.encoder(seqs, padding_mask, layer_output_hook)

        assert w2v2_layer_output is not None
        return w2v2_layer_output


class Wav2Vec2LayerOutputBuilder(Wav2Vec2Builder):
    def build_model(self) -> Wav2Vec2LayerOutputModel:
        """Build a model."""
        encoder_frontend = self.encoder_builder.build_frontend()

        encoder = self.encoder_builder.build_encoder()

        masker = self.build_masker()

        quantizer = self.build_quantizer()

        return Wav2Vec2LayerOutputModel(
            encoder_frontend,
            encoder,
            masker,
            quantizer,
            self.config.final_dim,
            self.config.final_proj_bias,
            self.config.num_distractors,
            self.config.logit_temp,
            self.config.diversity_loss_weight,
            device=self.device,
            dtype=self.dtype,
        )


def create_wav2vec2_layer_output_model(
    config: Wav2Vec2Config,
    device: Optional[Device] = None,
    dtype: Optional[DataType] = None,
) -> Wav2Vec2Model:
    """Create a wav2vec 2.0 model.

    :param config:
        The configuration to use.
    :param device:
        The device on which to initialize modules.
    :param dtype:
        The data type of module parameters and buffers.
    """
    encoder_builder = Wav2Vec2EncoderBuilder(config.encoder_config, device, dtype)

    return Wav2Vec2LayerOutputBuilder(
        config, encoder_builder, device, dtype
    ).build_model()


load_wav2vec2_layer_output_config = ModelConfigLoader[Wav2Vec2Config](
    asset_store, wav2vec2_archs
)

load_wav2vec2_layer_output_model = Wav2Vec2Loader(
    asset_store,
    download_manager,
    create_wav2vec2_layer_output_model,
    wav2vec2_archs,
    # `weight_norm` used in `Wav2Vec2PositionEncoder` does not support meta
    # initialization.
    use_meta=False,
)


if __name__ == "__main__":
    from fairseq2.data import Collater
    from fairseq2.memory import MemoryBlock
    from fairseq2.data.audio import AudioDecoder
    from pathlib import Path

    audio = "/large_experiments/seamless/ust/data/TTS/vocoder_training/audio_wavs/multi_spkr/eng/eng_LJSpeech-1.1_0/LJ003-0001.wav"
    out_layer_idx = 34
    device = torch.device("cuda:1")
    decode_audio = AudioDecoder(dtype=torch.float32, device=device)
    collate = Collater(pad_idx=2, pad_to_multiple=2)
    decoded_audio = None
    if isinstance(audio, str):
        with Path(audio).open("rb") as fb:
            block = MemoryBlock(fb.read())
        decoded_audio = decode_audio(block)
    src = collate(decoded_audio)["waveform"]

    x = torch.tensor(torch.load("/checkpoint/krs/x.pt"), device=device)
    print(f"After read audio: {x.sum()}, {x.shape}")
    x = x.unsqueeze(0)
    import torch.nn.functional as F

    x = F.layer_norm(x, x.shape)
    # batch.seqs = batch.seqs.view(1, -1)

    print(f"After layer norm: {x.sum()}, {x.shape}")
    model = load_wav2vec2_layer_output_model(
        "xlsr2_1b_v2", device=device, dtype=torch.float32
    )
    model.eval()
    batch = SequenceBatch(seqs=x, seq_lens=src["seq_lens"])
    out = model(batch, out_layer_idx)
    print(out.sum(), out.shape)
