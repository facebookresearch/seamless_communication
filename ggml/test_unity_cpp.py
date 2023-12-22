# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the license found in the
# MIT_LICENSE file in the root directory of this source tree.

import ctypes
import functools
from ctypes import c_void_p
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import fairseq2.nn
import fairseq2.nn.transformer
import numpy as np
import pytest
import torch
import torchaudio
from fairseq2.data.audio import WaveformToFbankConverter
from seamless_communication.inference.generator import SequenceGeneratorOptions
from fairseq2.models.wav2vec2.feature_extractor import Wav2Vec2FbankFeatureExtractor
from seamless_communication.inference.translator import Modality, Translator

import ggml
from ctypes_utils import NULLPTR, Ptr
from ggml import NativeObj
from ggml_convert import convert_model, read_layer_config
import requests

Ctx = ggml.ggml_context_p

UNITY_MODELS = Path(__file__).parent / "examples/unity/models"
FAIRSEQ2_CPP = Path(__file__).parent / "examples/unity/fairseq2.cpp"
UNITY_FLASH_ATTN = "\n# define UNITY_FLASH_ATTN 0\n" not in FAIRSEQ2_CPP.read_text()

DATA = Path(__file__).parent / "test_data"
LOCAL_AUDIO_SAMPLE_PATH = DATA / "LJ037-0171_sr16k.wav"
TEST_AUDIO_SAMPLE_URL = (
    "https://dl.fbaipublicfiles.com/seamless/tests/LJ037-0171_sr16k.wav"
)


MB = 1024 * 1024


@pytest.fixture(name="ctx")
def _ctx() -> Iterator[Ctx]:
    """Allocate a new context with 1024 MB of memory"""
    try:
        mem_size = 16 * MB
        memory = torch.zeros(mem_size, dtype=torch.uint8)
        ctx = ggml.ggml_init(
            params=ggml.ggml_init_params(
                mem_size=mem_size,
                mem_buffer=ctypes.c_void_p(memory.data_ptr()),
                no_alloc=True,
            )
        )
        with torch.inference_mode():
            yield ctx
    finally:
        ggml.ggml_free(ctx)


@functools.lru_cache()
def _load_g_model_once() -> NativeObj:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"
    if not model_file.exists():
        convert_model("seamlessM4T_medium", model_file)
    return ggml.load_fairseq2_ggml_file(model_file)


@pytest.fixture()
def g_model(ctx: Ctx) -> c_void_p:
    model = _load_g_model_once()
    ggml.lib.fairseq2_model_set_inference_ctx(model.ptr, ctx)
    return model.ptr


@functools.lru_cache(maxsize=1)
def load_translator() -> Translator:
    return Translator("seamlessM4T_medium", None, device=torch.device("cpu"))


def load_pt_model() -> Any:
    return load_translator().model


def download_sample_audio() -> Any:
    response = requests.get(TEST_AUDIO_SAMPLE_URL, stream=True)
    with open(DATA / "LJ037-0171_sr16k.wav", "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)


def test_convert_linear(tmp_path: Path) -> None:
    module = fairseq2.nn.Linear(16, 24, True)

    layer_config = read_layer_config(module)
    assert layer_config == {"input_dim": 16, "output_dim": 24}

    module_file = tmp_path / "module.ggml"
    convert_model(module, module_file)
    g_module = ggml.load_fairseq2_ggml_file(module_file)

    for k, v in layer_config.items():
        assert (
            ggml.fairseq2_model_layer_config_int(g_module.ptr, bytes(k, "ascii")) == v
        )

def test_convert_linear_fp16(tmp_path: Path, ctx: Ctx) -> None:
    pt_model = torch.nn.ModuleDict({"linear": fairseq2.nn.Linear(16, 24, True)})

    layer_config = read_layer_config(pt_model)
    assert layer_config == {"linear.input_dim": 16, "linear.output_dim": 24}

    ggml_file = tmp_path / "linear.ggml"
    convert_model(pt_model, ggml_file, fp16=True)
    assert ggml_file.stat().st_size < (16 * 24 + 24) * 2 * 1.5
    g_model = ggml.load_fairseq2_ggml_file(ggml_file)
    ggml.lib.fairseq2_model_set_inference_ctx(g_model.ptr, ctx)

    x = torch.empty((2, 5, 16))
    torch.nn.init.uniform_(x, -1, 1)
    y_exp = pt_model.linear(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("Linear", g_model.ptr, "linear", gx)
    ggml.build_and_compute(ctx, gy)
    y = ggml.to_numpy(gy)

    assert np.allclose(y_exp, y, atol=1e-3)


def test_causal_attention_mask(ctx: Ctx):
    x = torch.zeros((1, 10, 32))
    generator = fairseq2.nn.transformer.CausalAttentionMaskFactory()
    mask_exp = generator(x, x).materialize().numpy()

    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    ggml.build_and_compute(ctx, gmask)
    mask = ggml.to_numpy(gmask)

    assert mask_exp.shape == (10, 10)
    assert mask.shape == (10, 10)
    assert np.all(mask == mask_exp)

    x = x[:, :8, :]
    mask_exp = generator(x, x).materialize().numpy()
    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    ggml.build_and_compute(ctx, gmask)
    mask = ggml.to_numpy(gmask)

    assert mask_exp.shape == (8, 8)
    assert mask.shape == (8, 8)
    assert np.all(mask == mask_exp)


def test_LayerNorm_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    y_exp = pt_model.text_encoder.layers[0].ffn_layer_norm(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("LayerNorm", g_model, "text_encoder.layers.0.ffn_layer_norm", gx)
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def test_Linear_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    y_exp = pt_model.text_encoder.layers[0].ffn.inner_proj(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("Linear", g_model, "text_encoder.layers.0.ffn.inner_proj", gx)
    gf = ggml.build_and_compute(ctx, gy, dump="dot/test_Linear_forward.dot")

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def test_FeedForwardNetwork_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))  # (bs, seq_len, model_dim)
    torch.nn.init.uniform_(x, -1 / 32, 1 / 32)

    # Test FFN without LayerNorm
    pt_model = load_pt_model()
    y_exp = pt_model.text_encoder.layers[0].ffn(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward(
        "StandardFeedForwardNetwork", g_model, "text_encoder.layers.0.ffn", gx
    )
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


@pytest.mark.parametrize("lengths", [(11, 21), (21, 13)])
def test_MultiheadAttention_forward(
    ctx: Ctx, g_model: c_void_p, lengths: Tuple[int, int]
) -> None:
    x = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    # Note: we use different lengths for queries and keys,
    # this tests the implementation in decoding context too.
    # Note2: ggml_flash_attn requires that we have more keys than queries
    # qlen, klen = (11, 21) if flash_attn else (21, 13)
    qlen, klen = lengths
    xq = x[:, :qlen]
    xk = x[:, :klen]
    if qlen > klen and UNITY_FLASH_ATTN:
        pytest.skip(reason="flash_attn requires qlen > klen")

    gxq = ggml.from_numpy(ctx, xq.contiguous())
    ggml.ggml_set_name(gxq, b"xq")
    gxk = ggml.from_numpy(ctx, xk.contiguous())
    ggml.ggml_set_name(gxk, b"xk")
    ggml.ggml_set_no_alloc(ctx, True)
    gy = ggml.forward(
        "MultiheadAttention",
        g_model,
        "text_encoder.layers.0.self_attn",
        gxq,
        gxk,
        gxk,
        NULLPTR,  # TODO: tests with causal attention masks
    )
    gf = ggml.build_and_compute(ctx, gy, dump="dot/test_MultiheadAttention_forward")
    y = ggml.to_numpy(gy)
    nodes = ggml.nodes(gf)
    node_buffers = set(t.contents.data for t in nodes.values())

    pt_model = load_pt_model()
    self_attn = pt_model.text_encoder.layers[0].self_attn

    # If buffers are overlapping, reading node contents, can be misleading.
    overlap = len(node_buffers) < len(nodes)
    if not overlap:
        q_exp = self_attn._project_q(xq, None).numpy().reshape(2 * 16, qlen, 64)
        q = ggml.to_numpy(nodes[b"q"])
        assert q.shape == q_exp.shape
        assert np.allclose(q_exp, q, atol=1e-5)

        attn_weights_hook = fairseq2.nn.transformer.AttentionWeightStoreHook([])
        self_attn.register_attn_weight_hook(attn_weights_hook)

    y_exp = self_attn(xq, None, xk, None, xk).numpy()

    # with flash_attn we don't have attn_weights
    naive_attn = b"attn_weights" in nodes
    if naive_attn and not overlap:
        attn_weights = ggml.to_numpy(nodes[b"attn_weights"]).reshape(-1, 16, qlen, klen)
        [(_, attn_weights_exp)] = attn_weights_hook._storage
        attn_weights_exp = attn_weights_exp.numpy()
        assert attn_weights_exp.shape == attn_weights.shape
        # GGML is very agressively reducing small softmax weights to 0,
        # so the error isn't that small
        assert np.allclose(attn_weights_exp, attn_weights, atol=1e-3)
        # But the sums should be close to 1
        assert np.allclose(np.sum(attn_weights, axis=-1), np.ones((2, 16, qlen)))
        # And the maximum index should match the original ones.
        assert np.allclose(
            np.argmax(attn_weights_exp, axis=-1), np.argmax(attn_weights, axis=-1)
        )
    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-2 if naive_attn else 1e-4)


def test_MultiheadAttention_forward_self_attn_with_cache(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    attn = pt_model.text_decoder.layers[0].self_attn

    x = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    state_bag = fairseq2.nn.IncrementalStateBag(100)

    with ggml.fairseq2_kv_cache_alloc(g_model, 16 * MB, 2, 21):
        # Incremental decoding
        for t in range(3):
            xq = x[:, t : t + 1]

            gxq = ggml.from_numpy(ctx, xq.contiguous())
            ggml.ggml_set_name(gxq, b"xq")
            gy = ggml.forward(
                "MultiheadAttention",
                g_model,
                "text_decoder.layers.0.self_attn",
                gxq,
                gxq,
                gxq,
                None,  # type: ignore
            )
            gf = ggml.build_and_compute(
                ctx,
                gy,
                dump=f"dot/test_MultiheadAttention_forward_self_attn_with_cache_{t}.dot",
            )
            nodes = ggml.nodes(gf)
            gk_cache = ggml.to_numpy(
                nodes[b"text_decoder.layers.0.self_attn.k (step=%d)" % t]
            )
            assert gk_cache.shape == (2, t + 1, 1024)
            gk_cache = gk_cache.reshape(2, t + 1, 16, 64).transpose(0, 2, 1, 3)
            assert gk_cache.shape == (2, 16, t + 1, 64)

            y_exp = attn(xq, None, xq, None, xq, state_bag=state_bag).numpy()
            assert y_exp.shape == (2, 1, 1024)
            state = state_bag.get_state(attn, fairseq2.nn.transformer.AttentionState)
            state_bag.increment_step_nr()
            assert state is not None

            k_cache = state.get()[0].numpy()
            assert k_cache.shape == (2, 16, t + 1, 64)
            assert np.allclose(gk_cache, k_cache, atol=1e-3)

            y = ggml.to_numpy(gy)
            assert np.allclose(y, y_exp, atol=1e-2)


def test_MultiheadAttention_forward_cross_attn_with_cache(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    attn = pt_model.text_decoder.layers[0].encoder_decoder_attn

    x = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    state_bag = fairseq2.nn.IncrementalStateBag(100)

    with ggml.fairseq2_kv_cache_alloc(g_model, 16 * MB, 2, 21):
        # Incremental decoding, the keys come from the encoder, and don't change during decoding
        xk = x[:, :11]
        gxk = ggml.from_numpy(ctx, xk.contiguous(), name=b"xk")

        for t in range(3):
            xq = x[:, t : t + 1]

            gxq = ggml.from_numpy(ctx, xq.contiguous())
            ggml.ggml_set_name(gxq, b"xq")
            gy = ggml.forward(
                "MultiheadAttention",
                g_model,
                "text_decoder.layers.0.encoder_decoder_attn",
                gxq,
                gxk,
                gxk,
                None,  # type: ignore
            )
            gf = ggml.build_and_compute(
                ctx,
                gy,
                dump=f"dot/test_MultiheadAttention_forward_cross_attn_with_cache_{t}.dot",
            )
            y = ggml.to_numpy(gy)
            nodes = ggml.nodes(gf)
            leaves = ggml.leafs(gf)

            if t > 0:
                # the cache only appear in the graph during the second call
                state = state_bag.get_state(
                    attn, fairseq2.nn.transformer.AttentionState
                )
                assert state is not None
                assert np.allclose(
                    state.get()[0].transpose(1, 2).numpy(),
                    ggml.to_numpy(
                        nodes[
                            b"text_decoder.layers.0.encoder_decoder_attn.k_cache (view)"
                        ]
                    ),
                    atol=1e-3,
                )

            state_bag.increment_step_nr()
            y_exp = attn(xq, None, xk, None, xk, state_bag=state_bag).numpy()
            assert y_exp.shape == (2, 1, 1024)
            assert np.allclose(y, y_exp, atol=1e-2)


def test_StandardTransformerEncoderLayer_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    layer = pt_model.text_encoder.layers[0]

    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardTransformerEncoderLayer",
        g_model,
        "text_encoder.layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, padding_mask=None)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)


def test_StandardConformerEncoderLayer_forward(ctx: Ctx, g_model: c_void_p) -> None:
    pt_model = load_pt_model()
    x = torch.rand(1, 137, 1024)

    layer = pt_model.speech_encoder.inner.layers[0]
    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoderLayer",
        g_model,
        "speech_encoder.inner.layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, padding_mask=None)
    y_exp = y_exp.squeeze(0).numpy()
    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=2e-3)


def test_StandardConformerEncoderAdaptorLayer_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    torch.random.manual_seed(0)
    x = torch.rand(1, 137, 1024)
    layer = pt_model.speech_encoder.adaptor_layers[0]
    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoderAdaptorLayer",
        g_model,
        "speech_encoder.adaptor_layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, None)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=2e-3)


def test_StandardTransformerEncoder_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    padding_mask = fairseq2.nn.padding.PaddingMask(torch.tensor([21, 21]), 21)
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask.materialize())
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardTransformerEncoder",
        g_model,
        "text_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_encoder(x, padding_mask)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=5e-3)


def test_StandardConformerEncoder_forward(ctx: Ctx, g_model: c_void_p) -> None:
    pt_model = load_pt_model()
    if not LOCAL_AUDIO_SAMPLE_PATH.exists():
        download_sample_audio()
    wav, _ = torchaudio.load(LOCAL_AUDIO_SAMPLE_PATH)
    gx = ggml.from_numpy(ctx, wav * 2**15)  # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoder",
        g_model,
        "speech_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)

    cache = DATA / "test_StandardConformerEncoder_forward.npy"
    if not cache.exists():
        converter = WaveformToFbankConverter(
            num_mel_bins=80,
            waveform_scale=2**15,
            channel_last=True,
            standardize=True,
        )
        converter_input = {
            "waveform": wav.transpose(0, 1),
            "sample_rate": 16000.0,
            "format": -1,
        }

        pt_model = load_pt_model()
        speech_encoder_input = pt_model.speech_encoder_frontend(
            converter(converter_input)["fbank"].unsqueeze(0), None
        )[0]

        y_exp, _ = pt_model.speech_encoder(speech_encoder_input, None)
        y_exp = y_exp.numpy()
        np.save(cache, y_exp)
    else:
        y_exp = np.load(cache)

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-2)


def test_WaveformToFbank_forward(ctx: Ctx, g_model: c_void_p) -> None:
    converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
    )
    extractor = Wav2Vec2FbankFeatureExtractor(80, stride=2, sample_every_k=1)
    if not LOCAL_AUDIO_SAMPLE_PATH.exists():
        download_sample_audio()
    wav, _ = torchaudio.load(LOCAL_AUDIO_SAMPLE_PATH)
    gx = ggml.from_numpy(ctx, wav * 2**15)  # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")

    gy = ggml.forward("WaveformToFbank", g_model, "", gx)
    gf = ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    converter_input = {
        "waveform": wav.transpose(0, 1),
        "sample_rate": 16000.0,
        "format": -1,
    }
    y_exp, _ = extractor(converter(converter_input)["fbank"].unsqueeze(0), None)
    y_exp = y_exp.squeeze(0).numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=4e-3)  # reduce? error is from standardization


def test_PositionalEmbedding_forward(ctx: Ctx, g_model: c_void_p) -> None:
    seq = torch.zeros((4, 20, 1024), dtype=torch.float32)

    pos_encoder = fairseq2.nn.SinusoidalPositionEncoder(1024, 55, _legacy_pad_idx=1)
    y_exp = pos_encoder(seq, None)[0].numpy()

    gseq = ggml.from_numpy(ctx, seq[0].clone().numpy())
    ggml.ggml_set_name(gseq, b"seq")
    gy = ggml.forward(
        "PositionalEmbedding", g_model, "text_decoder_frontend.pos_encoder", gseq
    )
    gf = ggml.build_and_compute(ctx, gy, dump=True)
    y = ggml.to_numpy(gy)

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-6)


def test_PositionalEmbedding_forward_with_cache(ctx: Ctx, g_model: c_void_p) -> None:
    seq = torch.zeros((4, 20, 1024), dtype=torch.float32)
    pos_encoder = fairseq2.nn.SinusoidalPositionEncoder(1024, 55, _legacy_pad_idx=1)
    pos_encoder.eval()
    state_bag = fairseq2.nn.IncrementalStateBag(100)

    with ggml.fairseq2_kv_cache_alloc(g_model, 16 * MB, 2, 21):
        # Incremental decoding
        for t in range(20):
            gseq = ggml.from_numpy(ctx, seq[:, t : t + 1, :].numpy())
            ggml.ggml_set_name(gseq, b"seq")
            gy = ggml.forward(
                "PositionalEmbedding",
                g_model,
                "text_decoder_frontend.pos_encoder",
                gseq,
            )
            gf = ggml.build_and_compute(ctx, gy, dump=t == 1)
            y = ggml.to_numpy(gy)

            y_exp = pos_encoder(seq[:, t : t + 1, :], None, state_bag=state_bag).numpy()
            state_bag.increment_step_nr()
            assert y.shape == y_exp.shape
            assert np.allclose(y_exp, y, atol=1e-6)


def test_TransformerEmbeddingFrontend_forward(ctx: Ctx, g_model: c_void_p) -> None:
    seq = torch.arange(2 * 20).reshape(2, 20)
    seq[1, 15:] = 0  # padding for second sentence
    seq_len = torch.tensor([20, 15])
    gseq = ggml.from_numpy(ctx, seq.numpy().astype(np.int32))

    ggml.ggml_set_name(gseq, b"seq")
    gy = ggml.forward(
        "TransformerEmbeddingFrontend", g_model, "text_decoder_frontend", gseq
    )
    ggml.build_and_compute(ctx, gy)
    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_decoder_frontend(seq, seq_len)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-6)


def test_StandardTransformerDecoderLayer_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 13, 1024))
    encoder_out = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)
    torch.nn.init.uniform_(encoder_out, -1, 1)

    self_attn_mask = fairseq2.nn.transformer.CausalAttentionMaskFactory()(x, x)
    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gself_attn_mask = ggml.from_numpy(ctx, self_attn_mask.materialize().numpy())
    ggml.ggml_set_name(gself_attn_mask, b"self_attn_mask")
    genc = ggml.from_numpy(ctx, encoder_out)
    ggml.ggml_set_name(genc, b"encoder_out")
    gy = ggml.forward(
        "StandardTransformerDecoderLayer",
        g_model,
        "text_decoder.layers.0",
        gx,
        gself_attn_mask,
        genc,
        NULLPTR,  # TODO support padding mask,
    )
    ggml.build_and_compute(ctx, gy, dump=True)
    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_decoder.layers[0](x, None, encoder_output=encoder_out, self_attn_mask=self_attn_mask)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    # We still have some numerical imprecision
    assert np.allclose(y_exp, y, atol=0.1)
    assert np.sum(np.abs(y_exp-y) > 1e-2) < 20


def test_StandardTransformerDecoder_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 13, 1024))
    encoder_out = torch.empty((2, 21, 1024))
    padding_mask = fairseq2.nn.padding.PaddingMask(torch.tensor([13, 13]), 13)
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)
    torch.nn.init.uniform_(encoder_out, -1, 1)
    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask.materialize())
    ggml.ggml_set_name(gpad, b"padding_mask")
    genc = ggml.from_numpy(ctx, encoder_out)
    gy = ggml.forward(
        "StandardTransformerDecoder",
        g_model,
        "text_decoder",
        gx,
        None,  # TODO support padding mask,
        genc,
        None,
    )
    ggml.build_and_compute(ctx, gy)
    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_decoder(x, padding_mask, encoder_out, None)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-3)  # TODO: those tests are failing now


def test_s2tt(ctx: Ctx, g_model: c_void_p):
    if not LOCAL_AUDIO_SAMPLE_PATH.exists():
        download_sample_audio()
    src_audio_wav, _ = torchaudio.load(LOCAL_AUDIO_SAMPLE_PATH)
    sample_file = DATA / "LJ037-0171_sr16k.wav.trans"
    translator = load_translator()
    if not sample_file.exists():
        decoded_audio = {
            "waveform": src_audio_wav.t(),
            "sample_rate": 16000.0,
            "format": -1,
        }
        src = translator.collate(translator.convert_to_fbank(decoded_audio))["fbank"]

        text_out, _ = translator.get_prediction(
            translator.model,
            translator.text_tokenizer,
            translator.unit_tokenizer,
            src["seqs"],
            padding_mask=None,
            input_modality=Modality.SPEECH,
            output_modality=Modality.TEXT,
            tgt_lang="cmn",
            text_generation_opts=SequenceGeneratorOptions(),
            unit_generation_opts=None,
        )

        tgt_text = str(text_out[0])
        assert tgt_text == "专家的检查和证据使该委员会得出了结论,可能有五次枪击."
        with open(sample_file, "w") as f:
            f.write(tgt_text)

    with open(sample_file, "r") as exp:
        exp_tgt_text = exp.readlines()[0].strip()

    # Apply scale before sending into ggml!
    gx = ggml.from_numpy(ctx, src_audio_wav * 2**15)
    ggml.ggml_set_name(gx, b"x")
    encoder_out = ggml.forward(
        "StandardConformerEncoder",
        g_model,
        "speech_encoder",
        gx,
        NULLPTR,  # TODO support padding mask
    )
    gf = ggml.build_and_compute(ctx, encoder_out)

    beam_size = 5
    opts = ggml.SequenceGeneratorOptions(
        beam_size=beam_size,
        soft_max_seq_len_a=1,
        soft_max_seq_len_b=200,
        hard_max_seq_len=500,
    )
    job = ggml.SequenceGeneratorJob(
        opts=opts,
        prefix_seq=ggml.from_numpy(ctx, np.array([3, 256200]).astype(np.int32)),
        pad_idx=0,
        unk_idx=1,
        bos_idx=2,
        eos_idx=3,
    )
    result_ptr = ggml.generate_sequence(g_model, Ptr(job), encoder_out, NULLPTR, ctx)
    results = [result_ptr[i] for i in range(beam_size) if result_ptr[i].seq != None]
    tokens = [
        translator.text_tokenizer.model.index_to_token(id)
        for id in ggml.to_numpy(results[0].seq).tolist()
    ][2:-1]
    tokens = "".join(tokens).replace("▁", " ")[1:]
    assert tokens == exp_tgt_text
