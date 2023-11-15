import ggml
import ctypes
import torch
import pytest
import numpy as np
import torch
import fairseq2.nn
import fairseq2.nn.transformer
import logging
import sys
import functools
from typing import Tuple
from pathlib import Path
from ctypes_utils import Ptr
from ctypes import c_void_p
from typing import Any
from pathlib import Path
from typing import Iterator
from ggml import NativeObj
from ggml_convert import convert_model, read_layer_config
from seamless_communication.models.inference.translator import Translator, Modality
from fairseq2.data.audio import WaveformToFbankConverter
import torchaudio
from ctypes_utils import NULLPTR
from fairseq2.models.wav2vec2.feature_extractor import Wav2Vec2FbankFeatureExtractor

Ctx = ggml.ggml_context_p

UNITY_MODELS = Path(__file__).parent / "examples/unity/models"
CTX_PARAMS = ggml.ggml_init_params(mem_size=1024 * 1024 * 1024 * 5, mem_buffer=None)

FAIRSEQ2_CPP = Path(__file__).parent / "examples/unity/fairseq2.cpp"
UNITY_FLASH_ATTN = "\n# define UNITY_FLASH_ATTN 0\n" not in FAIRSEQ2_CPP.read_text()

DATA = Path(__file__).parent


@pytest.fixture(name="ctx")
def _ctx() -> Iterator[Ctx]:
    """Allocate a new context with 1024 MB of memory"""
    try:
        ctx = ggml.ggml_init(params=CTX_PARAMS)
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
    return Translator(
        "seamlessM4T_medium", "vocoder_36langs", torch.device("cpu"), torch.float32
    )


def load_pt_model() -> Any:
    return load_translator().model


def test_convert_linear(tmp_path: Path) -> None:
    module = fairseq2.nn.Linear(16, 24, True)

    layer_config = read_layer_config(module)
    assert layer_config == {"input_dim": 16, "output_dim": 24, "skip_init": False}

    module_file = Path("module.ggml")
    convert_model(module, module_file)
    g_module = ggml.load_fairseq2_ggml_file(module_file)

    for k, v in layer_config.items():
        assert (
            ggml.fairseq2_model_layer_config_int(g_module.ptr, bytes(k, "ascii")) == v
        )


def test_causal_attention_mask(ctx: Ctx):
    x = torch.zeros((1, 10, 32))
    generator = fairseq2.nn.transformer.CausalAttentionMaskGenerator()
    mask_exp = generator(x).numpy()

    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    mask = ggml.to_numpy(gmask)

    gf = ggml.ggml_build_forward(gmask)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    assert mask_exp.shape == (10, 10)
    assert mask.shape == (10, 10)
    assert np.all(mask == mask_exp)

    x = x[:, :8, :]
    mask_exp = generator(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    mask = ggml.to_numpy(gmask)

    gf = ggml.ggml_build_forward(gmask)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

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
    ggml.build_and_compute(ctx, gy)

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
    gxk = ggml.from_numpy(ctx, xk.contiguous())
    ggml.ggml_set_name(gxk, b"xk")
    gy = ggml.forward(
        "MultiheadAttention",
        g_model,
        "text_encoder.layers.0.self_attn",
        gxq,
        gxk,
        gxk,
        NULLPTR,  # TODO: tests with causal attention masks
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    pt_model = load_pt_model()
    self_attn = pt_model.text_encoder.layers[0].self_attn
    q_exp = self_attn.q_proj(xq).numpy()

    y = ggml.to_numpy(gy)
    nodes = ggml.nodes(gf)

    attn_weights_hook = fairseq2.nn.transformer.StoreAttentionWeights([])
    self_attn.register_attn_weight_hook(attn_weights_hook)

    y_exp = self_attn(xq, None, xk, xk).numpy()

    q = ggml.to_numpy(nodes[b"q"])
    assert q.shape == q_exp.shape
    assert np.allclose(q_exp, q, atol=1e-5)

    # with flash_attn we don't have attn_weights
    naive_attn = b"attn_weights" in nodes
    if naive_attn:
        attn_weights = ggml.to_numpy(nodes[b"attn_weights"])
        [attn_weights_exp] = attn_weights_hook._storage
        attn_weights_exp = attn_weights_exp.numpy()
        assert attn_weights_exp.shape == attn_weights.shape
        # GGML is very agressively reducing small softmax weights to 0,
        # so the error isn't that small
        assert np.allclose(attn_weights_exp, attn_weights, atol=1e-3)
        # But the sums should be close to 1
        assert np.allclose(np.sum(attn_weights, axis=-1), np.ones((2 * 16, qlen)))
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

    state_bag = fairseq2.nn.IncrementalStateBag()

    ggml.fairseq2_kv_cache_alloc(g_model, 2, 21)
    # Incremental decoding
    for t in range(3):
        xq = x[:, t : t + 1]
        y_exp = attn(xq, None, xq, xq, state_bag=state_bag).numpy()
        assert y_exp.shape == (2, 1, 1024)

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
        gf = ggml.ggml_build_forward(gy)
        ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

        nodes = ggml.nodes(gf)
        state = state_bag.get_state(
            attn, fairseq2.nn.transformer.MultiheadAttentionState
        )
        assert state is not None
        assert np.allclose(
            state.prev_k.numpy(),
            ggml.to_numpy(nodes[b"text_decoder.layers.0.self_attn.k_cache (step=%d)" % t]),
            atol=1e-3,
        )

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

    state_bag = fairseq2.nn.IncrementalStateBag()

    ggml.fairseq2_kv_cache_alloc(g_model, 2, 21)
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
        gf = ggml.ggml_build_forward(gy)
        ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
        y = ggml.to_numpy(gy)
        nodes = ggml.nodes(gf)
        leaves = ggml.leafs(gf)

        if t > 0:
            # the cache only appear in the graph during the second call
            state = state_bag.get_state(
                attn, fairseq2.nn.transformer.MultiheadAttentionState
            )
            assert state is not None
            assert np.allclose(
                state.prev_k.numpy(),
                ggml.to_numpy(nodes[b"text_decoder.layers.0.encoder_decoder_attn.k_cache"]),
                atol=1e-3,
            )

        y_exp = attn(xq, None, xk, xk, state_bag=state_bag).numpy()
        assert y_exp.shape == (2, 1, 1024)
        assert np.allclose(y, y_exp, atol=1e-2)


def test_StandardTransformerEncoderLayer_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    padding_mask = torch.ones((2, 21))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    pt_model = load_pt_model()
    layer = pt_model.text_encoder.layers[0]

    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask)
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardTransformerEncoderLayer",
        g_model,
        "text_encoder.layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, padding_mask)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)


def test_StandardConformerEncoderLayer_forward(ctx: Ctx, g_model: c_void_p) -> None:
    pt_model = load_pt_model()
    x = torch.load(
        "/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/dev/seqs_before_conformer_block.pt"
    )
    padding_mask = torch.ones((1, x.shape[1]))
    layer = pt_model.speech_encoder.inner.layers[0]
    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask[0])
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardConformerEncoderLayer",
        g_model,
        "speech_encoder.inner.layers.0",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, padding_mask)
    y_exp = y_exp.numpy()
    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=2e-3)


def test_StandardConformerEncoderAdaptorLayer_forward(
    ctx: Ctx, g_model: c_void_p
) -> None:
    pt_model = load_pt_model()
    x = torch.load(
        "/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/dev/seqs_before_adaptor.pt"
    )
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
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    y_exp, _ = layer(x, None)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=2e-3)


def test_StandardTransformerEncoder_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 21, 1024))
    padding_mask = torch.ones((2, 21))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask)
    ggml.ggml_set_name(gpad, b"padding_mask")
    gy = ggml.forward(
        "StandardTransformerEncoder",
        g_model,
        "text_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)

    pt_model = load_pt_model()
    y_exp, _ = pt_model.text_encoder(x, padding_mask)
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4)


def test_StandardConformerEncoder_forward(ctx: Ctx, g_model: c_void_p) -> None:
    pt_model = load_pt_model()
    wav, _ = torchaudio.load(DATA / "test.wav")
    gx = ggml.from_numpy(ctx, wav * 2**15)  # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoder",
        g_model,
        "speech_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

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

    y = ggml.to_numpy(gy)
    speech_encoder_input = pt_model.speech_encoder_frontend(
        converter(converter_input)["fbank"].unsqueeze(0), None
    )[0]

    y_exp, _ = pt_model.speech_encoder(speech_encoder_input, None)
    y_exp = y_exp.numpy()  # remove batch dimension

    assert y.shape == y_exp.shape
    assert np.allclose(
        y_exp, y, atol=1e-2
    )  # There are 10 elements in a 137*1024 tensor with error >1e-2


def test_WaveformToFbank_forward(ctx: Ctx, g_model: c_void_p) -> None:
    pt_model = load_pt_model()
    converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
    )
    extractor = Wav2Vec2FbankFeatureExtractor(80, 2, 1)
    wav, _ = torchaudio.load(
        "/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/test.wav"
    )
    gx = ggml.from_numpy(ctx, wav * 2**15)  # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")

    gy = ggml.forward("WaveformToFbank", g_model, "", gx)
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)
    converter_input = {
        "waveform": wav.transpose(0, 1),
        "sample_rate": 16000.0,
        "format": -1,
    }
    y_exp = extractor(converter(converter_input)["fbank"].unsqueeze(0), None)[0]
    y_exp = y_exp.numpy()

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=4e-3)  # reduce? error is from standardization


def test_causal_attention_mask(ctx: Ctx):
    x = torch.zeros((5, 10))
    generator = fairseq2.nn.transformer.CausalAttentionMaskGenerator()
    mask_exp = generator(x)

    gx = ggml.from_numpy(ctx, x)
    gmask = ggml.causal_attention_mask(ctx, gx)
    mask = ggml.to_numpy(gmask)

    assert mask_exp.shape == (10, 10)
    assert mask.shape == (10, 10)
    assert np.allclose(mask, mask_exp)


def test_PositionalEmbedding_forward(ctx: Ctx, g_model: c_void_p) -> None:
    seq = torch.zeros((4, 20, 1024), dtype=torch.float32)
    # this _legacy_pad_idx is suspicious. Shouldn't the model use 1 ? But
    # this is consistent with pt_model.text_decoder_frontend.pos_encoder._sin_offset
    pos_encoder = fairseq2.nn.SinusoidalPositionEncoder(1024, 55, _legacy_pad_idx=0)
    y_exp = pos_encoder(seq, None)[0].numpy()

    gseq = ggml.from_numpy(ctx, seq[0].numpy())
    ggml.ggml_set_name(gseq, b"seq")
    gy = ggml.forward(
        "PositionalEmbedding", g_model, "text_decoder_frontend.pos_encoder", gseq
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    y = ggml.to_numpy(gy)

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


def test_StandardTransformerDecoder_forward(ctx: Ctx, g_model: c_void_p) -> None:
    x = torch.empty((2, 13, 1024))
    encoder_out = torch.empty((2, 21, 1024))
    padding_mask = torch.ones((2, 13))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)
    torch.nn.init.uniform_(encoder_out, -1, 1)
    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask)
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
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-3)


def test_t2tt(ctx: Ctx, g_model: c_void_p) -> None:
    src_lang = "eng"
    src_text = "We are all in a yellow submarine."
    tgt_lang = "fra"
    sample_file = DATA / "sample_input.npz"
    beam_size = 2

    if not sample_file.exists():
        translator = load_translator()
        device = translator.device
        token_encoder = translator.text_tokenizer.create_encoder(
            task="translation", lang=src_lang, mode="source", device=device
        )
        src = translator.collate(token_encoder(src_text))

        text_out, _ = translator.get_prediction(
            translator.model,
            translator.text_tokenizer,
            translator.unit_tokenizer,
            src,
            input_modality=Modality.TEXT,
            output_modality=Modality.TEXT,
            tgt_lang=tgt_lang,
            beam_size=beam_size,
        )

        tgt_text = str(text_out.sentences[0])
        assert tgt_text == "Nous sommes tous dans un sous-marin jaune."
        hypotheses = [
            {
                "seq": h.seq.tolist(),
                "score": h.score.item(),
                "step_scores": h.step_scores.numpy(),
            }
            for h in text_out.generator_output.results[0]
        ]
        np.savez(
            sample_file,
            encoder_output=text_out.encoder_output.numpy(),
            encoder_padding_mask=text_out.encoder_padding_mask.numpy(),
            hypotheses=hypotheses,
        )

    # allow_pickle to load the hyp dicts
    text_out = np.load(sample_file, allow_pickle=True)
    encoder_out = ggml.from_numpy(ctx, text_out["encoder_output"])
    encoder_padding_mask = ggml.from_numpy(ctx, text_out["encoder_padding_mask"])
    prefix_seq = np.array(text_out["hypotheses"][0]["seq"][:2]).astype(np.int32)
    max_seq_len = max(len(h["seq"]) for h in text_out["hypotheses"])

    opts = ggml.SequenceGeneratorOptions(
        beam_size=beam_size,
        min_seq_len=1,
        soft_max_seq_len_a=1,
        soft_max_seq_len_b=200,
        hard_max_seq_len=int(max_seq_len * 1.5),
        len_penalty=1.0,
        unk_penalty=0.0,
        normalize_scores=True,
    )
    job = ggml.SequenceGeneratorJob(
        opts=opts,
        prefix_seq=ggml.from_numpy(ctx, prefix_seq),
        pad_idx=0,
        unk_idx=1,
        bos_idx=2,
        eos_idx=3,
    )

    result_ptr = ggml.generate_sequence(
        g_model, job, encoder_out, encoder_padding_mask, ctx
    )
    results = [result_ptr[i] for i in range(beam_size) if result_ptr[i].seq != None]

    assert len(results) == len(text_out["hypotheses"])
    for g_hyp, exp in zip(results, text_out["hypotheses"]):
        g_tokens = list(ggml.to_numpy(g_hyp.seq))
        g_step_scores = ggml.to_numpy(g_hyp.step_scores)
        assert g_tokens == exp["seq"]
        assert g_hyp.score == pytest.approx(exp["score"], rel=1e-2)
        # The score error is big, this may negatively impact the beam search.
        assert np.allclose(g_step_scores, exp["step_scores"], atol=0.1)


def test_s2tt(ctx: Ctx, g_model: c_void_p):
    src_audio_wav, _ = torchaudio.load(DATA / "test.wav")
    # translator = load_translator()
    # token_encoder = translator.text_tokenizer.create_encoder(
    #     task="translation"
    # )
    # decoded_audio = {
    #     "waveform": src_audio_wav.t(),
    #     "sample_rate": 16000.,
    #     "format": -1,
    # }
    # src = translator.collate(translator.convert_to_fbank(decoded_audio))["fbank"]

    # text_out, _ = translator.get_prediction(
    #     translator.model,
    #     translator.text_tokenizer,
    #     translator.unit_tokenizer,
    #     src,
    #     input_modality=Modality.SPEECH,
    #     output_modality=Modality.TEXT,
    #     tgt_lang="cmn",
    # )

    # tgt_text = str(text_out.sentences[0])
    # assert tgt_text == "大家好 , 世界无主题。"
    # tgt_tokens = text_out.generator_output.results[0][0].seq
    # score = text_out.generator_output.results[0][0].score.item()

    tgt_tokens = [
        3,
        256200,
        16991,
        249346,
        249725,
        146,
        25220,
        251069,
        249211,
        251148,
        253935,
        3,
    ]  # "大家好 , 世界无主题。"
    score = -1.606838583946228
    gx = ggml.from_numpy(
        ctx, src_audio_wav * 2**15
    )  # Apply scale before sending into ggml!
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "StandardConformerEncoder",
        g_model,
        "speech_encoder",
        gx,
        None,  # TODO support padding mask
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    encoder_out = gy

    opts = ggml.SequenceGeneratorOptions(
        beam_size=5,
        soft_max_seq_len_a=1,
        soft_max_seq_len_b=200,
        hard_max_seq_len=1000,
    )
    job = ggml.SequenceGeneratorJob(
        opts=opts,
        prefix_seq=ggml.from_numpy(ctx, np.array([3, 256200]).astype(np.int32)),
        pad_idx=0,
        unk_idx=1,
        bos_idx=2,
        eos_idx=3,
    )
    result_ptr = ggml.generate_sequence(g_model, job, encoder_out, NULLPTR, ctx)
    g_tokens = list(ggml.to_numpy(result_ptr[0].seq))
    assert g_tokens == tgt_tokens
