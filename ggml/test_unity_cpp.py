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
from pathlib import Path
from ctypes_utils import Ptr
from ctypes import c_void_p
from typing import Any
from pathlib import Path
from typing import Iterator
from ggml import NativeObj
from ggml_convert import convert_model
from seamless_communication.models.inference.translator import Translator, Modality

Ctx = ggml.ggml_context_p

UNITY_MODELS = Path(__file__).parent / "examples/unity/models"
CTX_PARAMS = ggml.ggml_init_params(mem_size=1024 * 1024 * 1024, mem_buffer=None)

FAIRSEQ2_CPP = Path(__file__).parent / "examples/unity/fairseq2.cpp"
UNITY_FLASH_ATTN = "\n# define UNITY_FLASH_ATTN 0\n" not in FAIRSEQ2_CPP.read_text()


@pytest.fixture(name="ctx")
def _ctx() -> Iterator[Ctx]:
    """Allocate a new context with 1024 MB of memory"""
    try:
        ctx = ggml.ggml_init(params=CTX_PARAMS)
        yield ctx
    finally:
        ggml.ggml_free(ctx)


@pytest.fixture(scope="module")
def g_model_once() -> Iterator[c_void_p]:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"
    if not model_file.exists():
        convert_model("seamlessM4T_medium", model_file)
    with ggml.load_unity_ggml_file(model_file) as model:
        yield model


@pytest.fixture()
def g_model(ctx: Ctx, g_model_once: c_void_p) -> c_void_p:
    ggml.lib.fairseq2_model_set_inference_ctx(g_model_once, ctx)
    return g_model_once


@pytest.fixture(scope="module")
def translator() -> Iterator[Any]:
    tr = Translator(
        "seamlessM4T_medium", "vocoder_36langs", torch.device("cpu"), torch.float32
    )
    with torch.inference_mode():
        yield tr


@pytest.fixture(scope="module")
def pt_model(translator: Translator) -> Any:
    model = translator.model
    print(model)
    return model


@pytest.mark.xfail(reason="TODO")
def test_hparams_code_is_up_to_date() -> None:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"

    hparams_header_file = model_file.with_suffix(".hparams.h")
    hparams_struct = hparams_header_file.read_text().strip()
    actual_code = (UNITY_MODELS.parent / "unity_model_loader.h").read_text()
    assert hparams_struct in actual_code


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


def test_LayerNorm_forward(ctx: Ctx, g_model: c_void_p, pt_model: Any) -> None:
    x = torch.empty((2, 21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    y_exp = pt_model.text_encoder.layers[0].ffn_layer_norm(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("LayerNorm", g_model, "text_encoder.layers.0.ffn_layer_norm", gx)
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def test_Linear_forward(ctx: Ctx, g_model: c_void_p, pt_model: Any) -> None:
    x = torch.empty((2, 21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    y_exp = pt_model.text_encoder.layers[0].ffn.inner_proj(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("Linear", g_model, "text_encoder.layers.0.ffn.inner_proj", gx)
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def test_FeedForwardNetwork_forward(ctx: Ctx, g_model: c_void_p, pt_model: Any) -> None:
    x = torch.empty((2, 21, 1024))  # (bs, seq_len, model_dim)
    torch.nn.init.uniform_(x, -1 / 32, 1 / 32)

    # Test FFN without LayerNorm
    y_exp = pt_model.text_encoder.layers[0].ffn(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward(
        "StandardFeedForwardNetwork", g_model, "text_encoder.layers.0.ffn", gx
    )
    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, atol=1e-5)


def _name(tensor: ggml.ggml_tensor_p) -> bytes:
    try:
        return tensor.contents.name  # type: ignore[no-any-return]
    except ValueError:
        return b"???"


def test_MultiheadAttention_forward(ctx: Ctx, g_model: c_void_p, pt_model: Any) -> None:
    x = torch.empty((2, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    self_attn = pt_model.text_encoder.layers[0].self_attn

    # Note: we use different lengths for queries and keys,
    # this tests the implementation in decoding context too.
    # Note2: ggml_flash_attn requires that we have more keys than queries
    gxq = ggml.from_numpy(ctx, x[:, :11, :].contiguous())
    gx = ggml.from_numpy(ctx, x)
    ggml.ggml_set_name(gx, b"x")
    gy = ggml.forward(
        "MultiheadAttention",
        g_model,
        "text_encoder.layers.0.self_attn",
        gxq,
        gx,
        gx,
        None,  # TODO: tests with causal attention masks
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    q_exp = self_attn.q_proj(x[:, :11, :]).numpy()

    y = ggml.to_numpy(gy)
    nodes = {}

    for i in range(gf.n_nodes):
        name = _name(gf.nodes[i])
        children = [_name(gf.nodes[i].contents.src[j]) for j in range(2)]
        print(name, f"op({gf.nodes[i].contents.op})", children)
        nodes[name] = ggml.to_numpy(gf.nodes[i])

    attn_weights_hook = fairseq2.nn.transformer.StoreAttentionWeights([])
    self_attn.register_attn_weight_hook(attn_weights_hook)

    y_exp = self_attn(x[:, :11, :], None, x, x).numpy()

    q = nodes[b"q"]
    assert q.shape == q_exp.shape
    assert np.allclose(q_exp, q, atol=1e-5)

    # with flash_attn we don't have attn_weights
    if not UNITY_FLASH_ATTN:
        attn_weights = nodes[b"attn_weights"]
        [attn_weights_exp] = attn_weights_hook._storage
        attn_weights_exp = attn_weights_exp.numpy()
        assert attn_weights_exp.shape == attn_weights.shape
        # GGML is very agressively reducing small softmax weights to 0,
        # so the error isn't that small
        assert np.allclose(attn_weights_exp, attn_weights, atol=1e-3)
        # But the sums should be close to 1
        assert np.allclose(np.sum(attn_weights, axis=-1), np.ones((2 * 16, 11)))
        # And the maximum index should match the original ones.
        assert np.allclose(
            np.argmax(attn_weights_exp, axis=-1), np.argmax(attn_weights, axis=-1)
        )
    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)


def test_StandardTransformerEncoderLayer_forward(
    ctx: Ctx, g_model: c_void_p, pt_model: Any
) -> None:
    x = torch.empty((1, 21, 1024))
    padding_mask = torch.ones((1, 21))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    layer = pt_model.text_encoder.layers[0]

    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask[0])
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
    y_exp = y_exp.squeeze(0).numpy()  # remove batch dimension

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)


def test_StandardTransformerEncoder_forward(
    ctx: Ctx, g_model: c_void_p, pt_model: Any
) -> None:
    x = torch.empty((1, 21, 1024))
    padding_mask = torch.ones((1, 21))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask[0])
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

    y_exp, _ = pt_model.text_encoder(x, padding_mask)
    y_exp = y_exp.squeeze(0).numpy()  # remove batch dimension

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4 if UNITY_FLASH_ATTN else 1e-2)


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


def test_TransformerEmbeddingFrontend_forward(
    ctx: Ctx, g_model: c_void_p, pt_model: Any
) -> None:
    seq = torch.arange(20).reshape(1, 20)
    seq_len = torch.tensor([20])
    gseq = ggml.from_numpy(ctx, seq[0].numpy().astype(np.int32))
    ggml.ggml_set_name(gseq, b"seq")
    gy = ggml.forward(
        "TransformerEmbeddingFrontend", g_model, "text_decoder_frontend", gseq
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    y = ggml.to_numpy(gy)

    y_exp, _ = pt_model.text_decoder_frontend(seq, seq_len)
    y_exp = y_exp.squeeze(0).numpy()  # remove batch dimension

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-6)


def test_StandardTransformerDecoder_forward(
    ctx: Ctx, g_model: c_void_p, pt_model: Any
) -> None:
    pytest.skip("foo")
    x = torch.empty((1, 13, 1024))
    encoder_out = torch.empty((1, 21, 1024))
    padding_mask = torch.ones((1, 13))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)
    torch.nn.init.uniform_(encoder_out, -1, 1)
    gx = ggml.from_numpy(ctx, x[0])
    ggml.ggml_set_name(gx, b"x")
    gpad = ggml.from_numpy(ctx, padding_mask[0])
    ggml.ggml_set_name(gpad, b"padding_mask")
    genc = ggml.from_numpy(ctx, encoder_out[0])
    gy = ggml.forward(
        "StandardTransformerDecoder",
        g_model,
        "text_decoder",
        gx,
        None,  # TODO support padding mask,
        genc,
        None,
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    y = ggml.to_numpy(gy)

    y_exp, _ = pt_model.text_decoder(x, padding_mask, encoder_out, None)
    y_exp = y_exp.squeeze(0).numpy()  # remove batch dimension

    assert y.shape == y_exp.shape
    assert np.allclose(y_exp, y, atol=1e-4)


def test_t2tt(ctx: Ctx, g_model: c_void_p):
    # def test_t2tt(ctx: Ctx, g_model: c_void_p, translator):
    # device = translator.device
    src_lang = "eng"
    src_text = "We are all in a yellow submarine."
    tgt_lang = "fra"
    # token_encoder = translator.text_tokenizer.create_encoder(
    #     task="translation", lang=src_lang, mode="source", device=device
    # )
    # src = translator.collate(token_encoder(src_text))

    # text_out, _ = translator.get_prediction(
    #     translator.model,
    #     translator.text_tokenizer,
    #     translator.unit_tokenizer,
    #     src,
    #     input_modality=Modality.TEXT,
    #     output_modality=Modality.TEXT,
    #     tgt_lang=tgt_lang,
    # )

    # tgt_text = str(text_out.sentences[0])
    # assert tgt_text == "Nous sommes tous dans un sous-marin jaune."
    # tgt_tokens = text_out.generator_output.results[0][0].seq
    # score = text_out.generator_output.results[0][0].score.item()
    # np.savez(
    #     Path(__file__).parent / "sample_input.npz",
    #     score=score,
    #     encoder_output=text_out.encoder_output.squeeze(0).numpy(),
    #     encoder_padding_mask=text_out.encoder_padding_mask.squeeze(0).numpy(),
    #     tgt_tokens=tgt_tokens.numpy(),
    # )

    text_out = np.load(Path(__file__).parent / "sample_input.npz")
    score = text_out["score"].item()

    tgt_tokens = list(text_out["tgt_tokens"])
    encoder_out = ggml.from_numpy(ctx, text_out["encoder_output"])
    encoder_padding_mask = ggml.from_numpy(ctx, text_out["encoder_padding_mask"])

    job = ggml.SequenceGeneratorJob()
    job.opts.beam_size = 1
    job.opts.min_seq_len = 1
    job.opts.soft_max_seq_len_a = 1
    job.opts.soft_max_seq_len_b = 200
    job.opts.hard_max_seq_len = int(len(tgt_tokens) * 1.5)
    job.opts.len_penalty = 1.0
    job.opts.unk_penalty = 0.0
    job.prefix_seq = ggml.from_numpy(ctx, text_out["tgt_tokens"].astype(np.int32)[:2])
    job.pad_idx = 0
    job.unk_idx = 1
    job.bos_idx = 2
    job.eos_idx = 3

    result = ggml.ggml_tensor()
    g_score = ggml.generate_sequence(
        g_model, job, encoder_out, encoder_padding_mask, ctypes.byref(result)
    )
    tokens = list(ggml.to_numpy(ctypes.pointer(result)))
    assert tokens == tgt_tokens
    assert g_score == pytest.approx(score)
