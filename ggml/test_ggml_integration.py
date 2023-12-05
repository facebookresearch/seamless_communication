import ctypes
import functools
import logging
import sys
from ctypes import c_void_p
from pathlib import Path
from typing import Any, Iterator, Tuple

import fairseq2.nn
import fairseq2.nn.transformer
import numpy as np
import pytest
import torch

import ggml
from ctypes_utils import Ptr
from ggml import NativeObj
from ggml_convert import convert_model

Ctx = ggml.ggml_context_p

UNITY_MODELS = Path(__file__).parent / "examples/unity/models"
CTX_PARAMS = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)


@pytest.fixture(name="ctx")
def _ctx() -> Iterator[Ctx]:
    """Allocate a new context with 16 MB of memory"""
    try:
        ctx = ggml.ggml_init(params=CTX_PARAMS)
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


def test_strides_works(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    assert ggml.strides(a) == np.ones((10,), dtype=np.float32).strides

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    assert ggml.strides(b) == np.ones((21, 11), dtype=np.float32).strides

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert ggml.strides(c) == np.ones((32, 22, 12), dtype=np.float32).strides


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
    at = ggml.to_numpy(gat)
    assert np.allclose(a.T, at)


def test_ggml_slice(ctx: Ctx) -> None:
    ga = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 10, 5)
    a = ggml.to_numpy(ga)
    a[...] = np.arange(50).reshape(5, 10).astype(dtype=np.float32)

    gs0 = ggml.ggml_slice(ctx, ga, 0, 3, 7)
    s0 = ggml.to_numpy(gs0)
    assert np.allclose(a[:, 3:7], s0)

    gs1 = ggml.ggml_slice(ctx, ga, 1, 2, 5)
    s1 = ggml.to_numpy(gs1)
    assert np.allclose(a[2:5, :], s1)


@pytest.mark.xfail(reason="to_numpy not implemented")
def test_ggml_transpose_and_slice(ctx: Ctx) -> None:
    ga = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 10, 5)
    a = ggml.to_numpy(ga)
    a[...] = np.arange(50).reshape(5, 10).astype(dtype=np.float32)

    gat = ggml.ggml_transpose(ctx, ga)
    gs0 = ggml.ggml_slice(ctx, gat, 0, 2, 5)
    s0 = ggml.to_numpy(gs0)
    assert np.allclose(a.T[:, 2:5], s0)

    gs1 = ggml.ggml_slice(ctx, gat, 1, 3, 7)
    s1 = ggml.to_numpy(gs1)
    assert np.allclose(a.T[3:7, :], s1)


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

    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y)


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_flatten(ctx: Ctx, ndim: int) -> None:
    shape = [11, 7, 5, 3][:ndim]  # Prime numbers to avoid surprises
    numel = functools.reduce(lambda a, b: a * b, shape, 1)
    x = torch.arange(numel, dtype=torch.float32).reshape(shape)
    for torch_dim in range(ndim - 1):
        ggml_dim = ndim - 1 - torch_dim
        n = x.shape[torch_dim + 1]

        gx = ggml.from_numpy(ctx, x)
        gx1 = ggml.ggml_flatten_1d(ctx, gx, ggml_dim - 1)
        gy = ggml.ggml_unflatten_1d(ctx, gx1, ggml_dim - 1, n)

        x1 = x.flatten(torch_dim, torch_dim + 1)
        y = x1.unflatten(torch_dim, (-1, n))
        assert y.shape == x.shape
        assert np.allclose(y.numpy(), x.numpy())
        assert x1.shape == ggml.shape(gx1)
        assert np.allclose(x1.numpy(), ggml.to_numpy(gx1))
        assert y.shape == ggml.shape(gy)
        assert np.allclose(y.numpy(), ggml.to_numpy(gy))


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
    # (H, slen, H_dim) -> (H, H_dim, slen)
    gv = ggml.from_numpy(ctx, v.transpose(1, 2).contiguous().numpy())
    assert ggml.shape(gv) == (num_heads, d_in, slen)
    gy = ggml.ggml_flash_attn(ctx, gq, gk, gv, True)
    gf = ggml.ggml_build_forward(gy)
    ggml.ggml_graph_compute_with_ctx(ctx, ctypes.pointer(gf), 1)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y)


@pytest.mark.parametrize("shape", [(5, 8, 4), (2, 5, 8, 4)])
def test_ggml_softmax_vs_torch(ctx: Ctx, shape: Tuple[int, ...]) -> None:
    x = torch.empty(shape)
    torch.nn.init.uniform_(x, -1, 1)
    y_exp = torch.softmax(x, dim=-1).numpy()

    gx = ggml.from_numpy(ctx, x.numpy())
    gy = ggml.ggml_soft_max(ctx, gx)

    ggml.build_and_compute(ctx, gy)

    y = ggml.to_numpy(gy)
    assert np.allclose(y_exp, y, rtol=1e-3)
    assert np.allclose(np.argmax(y_exp, axis=-1), np.argmax(y, axis=-1))


def test_can_return_hypothesis_ptr(ctx: Ctx) -> None:
    hyp_ptr = ggml._testing_return_hypothesis_ptr(ctx)

    hyp0, hyp1 = hyp_ptr[0], hyp_ptr[1]
    assert ggml.to_numpy(hyp0.seq).tolist() == [314]
    assert hyp0.score == pytest.approx(3.14)

    assert ggml.to_numpy(hyp1.seq).tolist() == [421]
    assert hyp1.score == pytest.approx(4.21)


@pytest.mark.parametrize("inplace", ["", "inplace"])
def test_set_2d(ctx: Ctx, inplace: bool):
    a = torch.empty((5, 3, 2))
    torch.nn.init.uniform_(a, -1, 1)
    b = torch.empty((3, 2))
    torch.nn.init.uniform_(b, -1, 1)
    a_original = a.clone()

    # make a copy of `a` before we modify it
    ga = ggml.from_numpy(ctx, a.clone().numpy())
    gb = ggml.from_numpy(ctx, b.numpy())
    a[3, ...] = b

    set_2d = ggml.ggml_set_2d_inplace if inplace else ggml.ggml_set_2d
    ga_updated = set_2d(ctx, ga, gb, ggml.nb(ga)[1], ggml.nb(ga)[2] * 3)
    ggml.build_and_compute(ctx, ga_updated)

    a_updated = ggml.to_numpy(ga if inplace else ga_updated)
    assert np.allclose(a.numpy(), a_updated)

    if not inplace:
        # When not using set_2d_inplace, the original tensor is unmodified.
        assert np.allclose(ggml.to_numpy(ga), a_original.numpy())
        assert ga.contents.data != ga_updated.contents.data
