import ggml
import ctypes
import torch
import pytest
import numpy as np
import torch
import fairseq2.nn
from typing import Any
from pathlib import Path
from typing import Iterator
from ggml import NativeObj
from ggml_convert import convert_model
from seamless_communication.models.unity import load_unity_model

Ctx = ggml.ggml_context_p

UNITY_MODELS = Path(__file__).parent / "examples/unity/models"
PARAMS_16MB = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)


@pytest.fixture(name="ctx")
def _ctx() -> Iterator[Ctx]:
    """Allocate a new context with 16 MB of memory"""
    try:
        ctx = ggml.ggml_init(params=PARAMS_16MB)
        yield ctx
    finally:
        ggml.ggml_free(ctx)


def test_ggml_bindings_work(ctx: Ctx) -> None:
    # Instantiate tensors
    x = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)
    b = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 1)

    # Use ggml operations to build a computational graph
    x2 = ggml.ggml_mul(ctx, x, x)
    f = ggml.ggml_add(ctx, ggml.ggml_mul(ctx, a, x2), b)

    gf = ggml.ggml_build_forward(f)

    # Set the input values
    ggml.ggml_set_f32(x, 2.0)
    ggml.ggml_set_f32(a, 3.0)
    ggml.ggml_set_f32(b, 4.0)

    # Compute the graph
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    # Get the output value
    output = ggml.ggml_get_f32_1d(f, 0)
    assert output == 16.0


def test_ggml_matmul(ctx: Ctx) -> None:
    # Instantiate tensors
    a = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 4, 2)
    x = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 4, 3)

    # Use ggml operations to build a computational graph
    y = ggml.ggml_mul_mat(ctx, a, x)
    assert ggml.shape(y) == (3, 2)
    gf = ggml.ggml_build_forward(y)

    # Set the input values
    ggml.ggml_set_f32(x, 0.0)
    for i in range(4 * 3):
        ggml.ggml_set_f32_1d(x, i, i)

    ggml.ggml_set_f32(a, 0.0)
    ggml.ggml_set_f32_1d(a, 1, 1.0)
    ggml.ggml_set_f32_1d(a, 7, 1.0)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    output = [[ggml.ggml_get_f32_1d(y, j * 2 + i) for j in range(3)] for i in range(2)]
    assert output == [[1, 5, 9], [3, 7, 11]]


def test_shape_works(ctx: Ctx) -> None:
    """GGML shape order convention is the reverse from numpy"""
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    assert ggml.shape(a) == (10,)

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    assert ggml.shape(b) == (21, 11)

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert ggml.shape(c) == (32, 22, 12)


def test_nb_works(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    assert ggml.nb(a) == (4, 40, 40, 40)

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F16, 11, 21)
    assert ggml.nb(b) == (2, 22, 462, 462)

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert ggml.nb(c) == (4, 48, 1056, 33792)


@pytest.mark.xfail(reason="TODO: fix strides")
def test_strides_works(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    assert ggml.strides(a) == np.ones((10,), dtype=np.float32).strides

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    assert ggml.strides(b) == np.ones((11, 21), dtype=np.float32).strides

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert ggml.strides(c) == np.ones((12, 22, 32), dtype=np.float32).strides


def test_to_numpy_works_with_f32(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    na = ggml.to_numpy(a)
    for i in range(10):
        ggml.ggml_set_f32_1d(a, i, i)
    assert na[5] == 5
    assert np.allclose(na, np.array(range(10), dtype=np.float32))
    ggml.ggml_set_f32_1d(a, 5, -1.5)
    assert na[5] == -1.5

    # Note: GGML order of dims is reversed wrt numpy shapes
    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    for i in range(11 * 21):
        ggml.ggml_set_f32_1d(b, i, i)
    nb = ggml.to_numpy(b)
    # assert nb.shape == (21, 11)
    assert nb[0, 5] == 5
    assert nb[3, 5] == 11 * 3 + 5
    assert np.allclose(
        nb, np.array(range(11 * 21), dtype=np.float32).reshape(ggml.shape(b))
    )
    ggml.ggml_set_f32_1d(b, 11 * 3 + 5, -1.5)
    assert nb[3, 5] == -1.5

    sum_rows = ggml.ggml_sum_rows(ctx, b)
    gf = ggml.ggml_build_forward(sum_rows)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    np_sum_rows = np.sum(nb, axis=-1, keepdims=True)
    assert np_sum_rows.shape == ggml.shape(sum_rows)
    for i in range(11):
        assert np_sum_rows[i] == ggml.ggml_get_f32_1d(sum_rows, i)

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    for i in range(12 * 22 * 32):
        ggml.ggml_set_f32_1d(c, i, i)
    nc = ggml.to_numpy(c)
    assert ggml.shape(c) == (32, 22, 12)
    assert nc[3, 5, 11] == 22 * 12 * 3 + 12 * 5 + 11
    assert np.allclose(
        nc, np.array(range(12 * 22 * 32), dtype=np.float32).reshape(ggml.shape(c))
    )
    ggml.ggml_set_f32_1d(c, 22 * 12 * 3 + 12 * 5 + 11, -1.5)
    assert nc[3, 5, 11] == -1.5


def test_from_numpy_works_with_f32(ctx: Ctx) -> None:
    a = np.random.normal(size=(10,)).astype(dtype=np.float32)
    ga = ggml.from_numpy(ctx, a)
    assert ggml.shape(ga) == (10,)
    assert ggml.nb(ga) == ggml.nb(ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10))
    assert np.allclose(a, ggml.to_numpy(ga))

    a = np.random.normal(size=(11, 21)).astype(dtype=np.float32)
    ga = ggml.from_numpy(ctx, a)
    assert ggml.shape(ga) == (11, 21)
    assert ggml.nb(ga) == ggml.nb(
        ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, *a.shape[::-1])
    )
    assert np.allclose(a, ggml.to_numpy(ga))

    a = np.random.normal(size=(12, 22, 32)).astype(dtype=np.float32)
    ga = ggml.from_numpy(ctx, a)
    assert ggml.shape(ga) == (12, 22, 32)
    assert ggml.nb(ga) == ggml.nb(
        ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, *a.shape[::-1])
    )
    assert np.allclose(a, ggml.to_numpy(ga))


def test_to_numpy_works_with_f16(ctx: Ctx) -> None:
    # We explicitly fill the tensor otherwise they might have non-zero values in them.
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F16, 10)
    na = ggml.to_numpy(a)
    ggml.ggml_set_f32(a, 2.14)
    assert np.allclose(na, np.ones((10,), dtype=np.float16) * 2.14)
    ggml.ggml_set_f32(a, 4.28)
    assert np.allclose(na, np.ones((10,), dtype=np.float16) * 4.28)

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F16, 11, 21)
    nb = ggml.to_numpy(b)
    ggml.ggml_set_f32(b, 4.18)
    assert np.allclose(nb, np.ones((21, 11), dtype=np.float16) * 4.18)
    ggml.ggml_set_f32(b, 5.12)
    assert np.allclose(nb, np.ones((21, 11), dtype=np.float16) * 5.12)

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F16, 12, 22, 32)
    nc = ggml.to_numpy(c)
    ggml.ggml_set_f32(c, 3.16)
    assert np.allclose(nc, np.ones((32, 22, 12), dtype=np.float16) * 3.16)
    ggml.ggml_set_f32(c, 5.08)
    assert np.allclose(nc, np.ones((32, 22, 12), dtype=np.float16) * 5.08)


def test_from_numpy_works_with_f16(ctx: Ctx) -> None:
    a = np.random.normal(size=(10,)).astype(dtype=np.float16)
    ga = ggml.from_numpy(ctx, a)
    assert np.allclose(a, ggml.to_numpy(ga))
    a = np.random.normal(size=(11, 21)).astype(dtype=np.float16)
    ga = ggml.from_numpy(ctx, a)
    assert np.allclose(a, ggml.to_numpy(ga))
    a = np.random.normal(size=(12, 22, 32)).astype(dtype=np.float16)
    ga = ggml.from_numpy(ctx, a)
    assert np.allclose(a, ggml.to_numpy(ga))


def test_to_numpy_works_with_transposed(ctx: Ctx) -> None:
    ga = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 10, 5)
    a = ggml.to_numpy(ga)
    a[...] = np.arange(50).reshape(5, 10).astype(dtype=np.float32)

    gat = ggml.ggml_transpose(ctx, ga)

    gf = ggml.ggml_build_forward(ga)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    at = ggml.to_numpy(gat)

    assert np.allclose(a.T, at)


def test_ning_model_load(ctx: Ctx) -> None:
    pytest.skip("borken")
    model, vocab = ggml.unity_model_load(UNITY_MODELS / "unity-large/ggml-model.bin")
    print(model, vocab)

    example = ggml.from_file(
        ctx, UNITY_MODELS / "unity-large/seqs_before_conformer_block.bin", (1024, 137)
    )

    with ggml.MeasureArena() as arena:
        graph = ggml.unity_audio_encoder_graph(model, example)
        # TODO: why the extra memory ?
        mem_size = ggml.ggml_allocr_alloc_graph(arena, graph) + ggml.GGML_MEM_ALIGN

    with ggml.FixedSizeArena(mem_size) as allocr:
        print(
            f"unity_audio_encoder_graph: compute buffer size: {mem_size/1024/1024} MB"
        )

        eval_res_ptr = ggml.unity_eval(allocr, model, example, 1)
        eval_res = eval_res_ptr.contents
        inpL = ggml.to_numpy(eval_res.nodes[eval_res.n_nodes - 1])
        expected_raw = "-0.1308,0.0346,-0.2656,0.2873,-0.0104,0.0574,0.4033,-0.1125,-0.0460,-0.0496"
        expected = map(float, expected_raw.split(","))
        assert np.allclose(inpL[0, :10], list(expected), atol=1e-4)


@pytest.fixture(scope="module")
def g_model_once() -> Iterator[ctypes.c_void_p]:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"
    if not model_file.exists():
        convert_model("seamlessM4T_medium", model_file)
    with ggml.load_unity_ggml_file(model_file) as model:
        yield model


@pytest.fixture()
def g_model(ctx: Ctx, g_model_once: ctypes.c_void_p) -> ctypes.c_void_p:
    ggml.lib.fairseq2_model_set_inference_ctx(g_model_once, ctx)
    return g_model_once


@pytest.fixture(scope="module")
def pt_model() -> Iterator[Any]:
    model = load_unity_model("seamlessM4T_medium")
    print(model)
    model.eval()
    with torch.inference_mode():
        yield model


@pytest.mark.xfail(reason="TODO")
def test_hparams_code_is_up_to_date() -> None:
    model_file = Path(__file__).parent / "seamlessM4T_medium.ggml"

    hparams_header_file = model_file.with_suffix(".hparams.h")
    hparams_struct = hparams_header_file.read_text().strip()
    actual_code = (UNITY_MODELS.parent / "unity_model_loader.h").read_text()
    assert hparams_struct in actual_code


def test_numpy_mul_mat(ctx: Ctx) -> None:
    slen, d_in, d_out = (5, 4, 2)
    # torch.nn and fairseq2.nn assumes (seq_len, dim) to represent inputs,
    x = np.zeros((slen, d_in), dtype=np.float32)  # (seq_len, dim_in)
    x[0, :] = [1, 1 / 3, 0, 0]

    weight = np.eye(d_out, d_in, dtype=np.float32)
    weight[1, 1] = 1
    # assert weight.shape == (d_out, d_in) # (dim_out, dim_in)
    y_exp = x @ weight.T  # (seq_len, dim_out)

    gx = ggml.from_numpy(ctx, x)  # (dim_in, seq_len)
    gw = ggml.from_numpy(ctx, weight)  # (dim_in, dim_out)
    # gb = ggml.from_numpy(ctx, linear.bias.numpy())  # (dim_out)
    # GGML linear impl
    assert ggml.ggml_can_mul_mat(gw, gx)
    # gy = ggml.ggml_add(ctx, ggml.ggml_mul_mat(ctx, gw, gx), gb)  # (dim_out, seq_len)
    gy = ggml.ggml_mul_mat(ctx, gw, gx)  # (dim_out, seq_len)

    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gf.nodes[gf.n_nodes - 1])
    assert np.allclose(y_exp, y)


@torch.no_grad()
def test_torch_spda_vs_ggml_flash_attn(ctx: Ctx) -> None:
    slen, d_in, num_heads = (5, 4, 2)
    torch.random.manual_seed(0)
    q = torch.zeros((num_heads, slen, d_in))
    torch.nn.init.uniform_(q, -1, 1)
    k = torch.zeros((num_heads, slen, d_in))
    torch.nn.init.uniform_(k, -1, 1)
    v = torch.zeros((num_heads, slen, d_in))
    torch.nn.init.uniform_(v, -1, 1)

    # Note: we are using x for both keys and queries, so every position
    # attends mostly to itself, hence y_exp looks a bit like arange(slen)
    y_exp = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    y_exp = y_exp.numpy()
    gq = ggml.from_numpy(ctx, q.numpy())
    gk = ggml.from_numpy(ctx, k.numpy())
    # ggml flash attention expect a different order of axis for v:
    gv = ggml.from_numpy(ctx, v.transpose(1, 2).contiguous().numpy())
    assert ggml.shape(gv) == (num_heads, d_in, slen)
    gy = ggml.ggml_flash_attn(ctx, gq, gk, gv, True)
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y)


def test_forward_ffn(ctx: Ctx, g_model: NativeObj, pt_model: Any) -> None:
    x = torch.empty((21, 1024))  # (seq_len, model_dim)
    torch.nn.init.uniform_(x, -1 / 32, 1 / 32)

    # Test FFN without LayerNorm
    y_exp = pt_model.text_encoder.layers[0].ffn(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward(
        "StandardFeedForwardNetwork", g_model, "text_encoder.layers.0.ffn", gx
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gf.nodes[gf.n_nodes - 1])
    assert np.allclose(y_exp, y, rtol=2e-2, atol=1e-4)


def test_forward_layer_norm(ctx: Ctx, g_model: NativeObj, pt_model: Any) -> None:
    x = torch.empty((21, 1024))
    torch.nn.init.uniform_(x, -1, 1)

    y_exp = pt_model.text_encoder.layers[0].ffn_layer_norm(x).numpy()
    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward("LayerNorm", g_model, "text_encoder.layers.0.ffn_layer_norm", gx)
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gf.nodes[gf.n_nodes - 1])
    assert np.allclose(y_exp, y, rtol=1e-3, atol=1e-4)


def test_forward_self_attn(ctx: Ctx, g_model: NativeObj, pt_model: Any) -> None:
    x = torch.empty((1, 21, 1024))
    torch.random.manual_seed(0)
    torch.nn.init.uniform_(x, -1, 1)

    self_attn = pt_model.text_encoder.layers[0].self_attn
    # Replace spda by just returning queries
    # TODO: implement spda
    # self_attn.spda = lambda *qkv, **kwargs: qkv[0]


    gx = ggml.from_numpy(ctx, x)
    gy = ggml.forward(
        "MultiheadAttention",
        g_model,
        "text_encoder.layers.0.self_attn",
        gx,
        gx,
        gx,
        None,
    )
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)
    y = ggml.to_numpy(gy)
    names = "ql,q,qt,qp,kl,k,kt,kp,vl,v,vt,vp,v_cont,attn,attn_p,attn_cont,attn_reshape,outl,out"
    assert gf.n_nodes == len(names.split(","))
    gf_nodes = {}
    for i, name in enumerate(names.split(",")):
        mid = ggml.to_numpy(gf.nodes[i])
        # print(name, mid.shape, mid)
        gf_nodes[name] = mid

    breakpoint()
    y_exp = self_attn(x, None, x, x).numpy()
    y_exp = y_exp.squeeze(0)  # remove batch dimension

    assert y.shape == y_exp.shape
    abs_diff = np.max(np.abs(y - y_exp))
    assert np.allclose(y_exp, y)
