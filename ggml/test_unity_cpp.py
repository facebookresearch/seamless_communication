import ggml
import ctypes
import torch
import pytest
import numpy as np
from pathlib import Path
from typing import Iterator
from ggml import NativeObj

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


def test_shape_works(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    assert ggml.shape(a) == (10,)

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    assert ggml.shape(b) == (11, 21)

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert ggml.shape(c) == (12, 22, 32)


@pytest.mark.xfail(
    reason="TODO: understand diff between ggml strides and numpy strides"
)
def test_strides_works(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    assert ggml.strides(a) == np.ones((10,), dtype=np.float32).strides

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    assert ggml.strides(b) == np.ones((11, 21), dtype=np.float32).strides

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert ggml.strides(c) == np.ones((12, 22, 32), dtype=np.float32).strides


def test_to_numpy_works_with_f32(ctx: Ctx) -> None:
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F32, 10)
    a = ggml.ggml_set_f32(a, 2.14)
    assert np.allclose(ggml.to_numpy(a), np.ones((10,)) * 2.14)
    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F32, 11, 21)
    assert np.allclose(ggml.to_numpy(b), np.zeros((11, 21)))
    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F32, 12, 22, 32)
    assert np.allclose(ggml.to_numpy(c), np.zeros((12, 22, 32)))


def test_from_numpy_works_with_f32(ctx: Ctx) -> None:
    a = np.random.normal(size=(10,)).astype(dtype=np.float32)
    ga = ggml.from_numpy(ctx, a)
    assert np.allclose(a, ggml.to_numpy(ga))
    a = np.random.normal(size=(11, 21)).astype(dtype=np.float32)
    ga = ggml.from_numpy(ctx, a)
    assert np.allclose(a, ggml.to_numpy(ga))
    a = np.random.normal(size=(12, 22, 32)).astype(dtype=np.float32)
    ga = ggml.from_numpy(ctx, a)
    assert np.allclose(a, ggml.to_numpy(ga))


def test_to_numpy_works_with_f16(ctx: Ctx) -> None:
    # We explicitly fill the tensor otherwise they might have non-zero values in them.
    a = ggml.ggml_new_tensor_1d(ctx, ggml.GGML_TYPE_F16, 10)
    a = ggml.ggml_set_f32(a, 2.14)
    assert np.allclose(ggml.to_numpy(a), np.ones((10,), dtype=np.float16) * 2.14)

    b = ggml.ggml_new_tensor_2d(ctx, ggml.GGML_TYPE_F16, 11, 21)
    b = ggml.ggml_set_f32(b, 4.18)
    assert np.allclose(ggml.to_numpy(b), np.ones((11, 21), dtype=np.float16) * 4.18)

    c = ggml.ggml_new_tensor_3d(ctx, ggml.GGML_TYPE_F16, 12, 22, 32)
    c = ggml.ggml_set_f32(c, 3.16)
    assert np.allclose(ggml.to_numpy(c), np.ones((12, 22, 32), dtype=np.float16) * 3.16)


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


def test_unity_model_load(ctx: Ctx) -> None:
    model, vocab = ggml.unity_model_load(
        UNITY_MODELS / "unity-large/ggml-model.bin"
    )
    print(model, vocab)

    example = ggml.from_file(ctx, UNITY_MODELS / "unity-large/seqs_before_conformer_block.bin", (1024, 137))

    with ggml.MeasureArena() as arena:
        graph = ggml.unity_audio_encoder_graph(model, example)
        # TODO: why the extra memory ?
        mem_size = ggml.ggml_allocr_alloc_graph(arena.ptr, graph) + ggml.GGML_MEM_ALIGN

    with ggml.FixedSizeArena(mem_size) as allocr:
        print(f"unity_audio_encoder_graph: compute buffer size: {mem_size/1024/1024} MB")

        eval_res_ptr = ggml.unity_eval(allocr, model, example, 1)
        eval_res = eval_res_ptr.contents
        inpL = ggml.to_numpy(eval_res.nodes[eval_res.n_nodes - 1])
        expected_raw = "-0.1308,0.0346,-0.2656,0.2873,-0.0104,0.0574,0.4033,-0.1125,-0.0460,-0.0496"
        expected = map(float, expected_raw.split(","))
        assert np.allclose(inpL[0, :10], list(expected), atol=1e-4)


# def test_unity_model_load2(ctx: Ctx) -> None:
#     model = ggml.unity_model_load(
#         UNITY_MODELS / "unity-large/ggml-model.bin"
#     )
#     print(model, vocab)
#
#
