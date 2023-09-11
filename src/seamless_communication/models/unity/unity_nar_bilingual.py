# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.mbart import mbart_archs
from seamless_communication.models.unity.builder import UnitYConfig, unity_arch
from seamless_communication.models.unity.loader import load_unity_model
from seamless_communication.models.unity.t2u_builder import (
    UnitYT2UConfig,
    NARDecoderConfig,
    unity_t2u_archs,
    unity_t2u_arch,
)
from seamless_communication.models.unity.nar_decoder_frontend import (
    NARDecoderFrontendConfig,
    VariancePredictorConfig,
)
from fairseq2.models.wav2vec2 import Wav2Vec2EncoderConfig
from fairseq2.nn.transformer import TransformerNormOrder

import torch


layer_descs = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

wav2vec2_encoder_config = Wav2Vec2EncoderConfig(
    model_dim=1024,
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
    pos_encoder_type="relative",
    pos_encoder_depth=1,
    pos_conv_kernel_size=128,
    num_pos_conv_groups=16,
    use_conformer=True,
    num_encoder_layers=24,
    num_encoder_attn_heads=16,
    ffn_inner_dim=4096,
    dropout_p=0.1,
    attn_dropout_p=0.0,
    layer_drop_p=0.0,
    norm_order=TransformerNormOrder.POST,
    depthwise_conv_kernel_size=31,
)


@unity_t2u_arch("nar_bilingual")
def _nar_bilingual_t2u() -> UnitYT2UConfig:
    duration_predictor_config = VariancePredictorConfig(
        var_pred_hidden_dim=256,
        var_pred_kernel_size=3,
        var_pred_dropout=0.5,
    )

    decoder_frontend_config = NARDecoderFrontendConfig(
        subword_to_unit_upsampling_type="hard",
        duration_predictor_config=duration_predictor_config,
        pitch_predictor_config=None,
        energy_predictor_config=None,
        layer_norm=False,
    )

    nar_decoder_config = NARDecoderConfig(
        text_tokenizer_type="mbart",
        model_name_or_card="unity_nar_bilingual",
        decoder_frontend_config=decoder_frontend_config,
        char_vocabulary_size=246,
        char_pad_idx=1,
        conv1d_kernel_size=7,
        conv1d_inner_dim=1024,
        conv1d_dropout_p=0.1,
    )

    return UnitYT2UConfig(
        model_dim=1024,
        unit_max_seq_len=2048,
        unit_vocabulary_size=1007,
        unit_pad_idx=1,
        num_encoder_layers=2,
        num_decoder_layers=2,
        nar_decoder_config=nar_decoder_config,
        num_encoder_attn_heads=16,
        num_decoder_attn_heads=16,
        ffn_inner_dim=4096,
        dropout_p=0.1,
    )


@unity_arch("unity_nar_bilingual")
def _nar_bilingual() -> UnitYConfig:
    mbart_config = mbart_archs.get_config("base")
    mbart_config.pos_encoder_type = "sinusoidal"
    mbart_config.frontend_layernorm = False

    t2u_config = unity_t2u_archs.get_config("nar_bilingual")

    return UnitYConfig(
        model_dim=1024,
        w2v2_encoder_config=wav2vec2_encoder_config,
        mt_model_config=mbart_config,
        t2u_config=t2u_config,
        use_text_encoder=False,
        use_conformer_adaptor=False,
        num_adaptor_layers=1,
        adaptor_kernel_size=3,
        adaptor_stride=2,
        adaptor_layer_norm=False,
        adaptor_dropout_p=0.1,
    )


def test_tokenizers(model_name_or_card):
    from seamless_communication.models.unity.char_tokenizer import (
        load_unity_char_tokenizer,
    )
    from fairseq2.models.mbart.loader import mBartTokenizerLoader
    from seamless_communication.assets import asset_store
    from fairseq2.assets import download_manager
    from fairseq2.data.text.sentencepiece import vocabulary_from_sentencepiece

    text_tokenizer = mBartTokenizerLoader(asset_store, download_manager)(
        model_name_or_card
    )

    print(vocabulary_from_sentencepiece(text_tokenizer.model))

    char_tokenizer = load_unity_char_tokenizer(model_name_or_card)

    print(vocabulary_from_sentencepiece(char_tokenizer.model))

    for i in range(6):
        print(i, text_tokenizer.model.index_to_token(i))

    for i in range(65530, 65539):
        print(i, text_tokenizer.model.index_to_token(i))

    sample_tokens = torch.tensor(
        [
            65536,
            0,
            655,
            9692,
            2049,
            19,
            22,
            146,
            31,
            29678,
            13,
            1845,
            17277,
            4120,
            5,
            56,
            22,
            15,
            5277,
            4,
            2,
        ],
        dtype=torch.int32,
    )

    print(f"{sample_tokens=}")
    text_token_decoder = text_tokenizer.create_decoder()
    decoded_str = str(text_token_decoder(sample_tokens)[0])
    print(f"{decoded_str=}")

    text_token_encoder = text_tokenizer.create_encoder(mode="target")
    print("prefix_indices: ", text_token_encoder.prefix_indices)
    print("suffix_indices: ", text_token_encoder.suffix_indices)
    encoded_tokens = text_token_encoder(decoded_str)
    print(f"{encoded_tokens=}")
    round_trip_str = str(text_token_decoder(encoded_tokens)[0])
    print(f"{round_trip_str=}")

    assert round_trip_str == decoded_str

    characters = "".join(ch for w in decoded_str.split() for ch in list(w))
    print(characters)

    char_encoder = char_tokenizer.create_encoder()
    char_tokens = char_encoder(decoded_str)
    print(char_tokens)

    char_decoder = char_tokenizer.create_decoder()
    decoded_char_str = str(char_decoder(char_tokens)[0])
    print(decoded_char_str)

    assert decoded_char_str == decoded_str


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    model_name_or_card = "unity_nar_bilingual"
    # test_tokenizers(model_name_or_card)

    model = load_unity_model(model_name_or_card, device=device, dtype=dtype)
