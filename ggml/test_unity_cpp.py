import ggml
import ctypes


def test_ggml_bindings_work() -> None:
    # Allocate a new context with 16 MB of memory
    params = ggml.ggml_init_params(mem_size=16 * 1024 * 1024, mem_buffer=None)
    ctx = ggml.ggml_init(params=params)

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

    # Free the context
    ggml.ggml_free(ctx)


def test_unity_model_load() -> None:
    model, vocab = ggml.unity_model_load(
        "examples/unity/models/unity-large/ggml-model.bin"
    )
    print(model, vocab)
