#include "ggml.h"
#include "fairseq2.h"

/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc() {
    // pre-allocate some memory to write hyperparameters and tensors pointers
    auto* model = new fairseq2_model;
    model->hparams = new std::uint8_t[8 * 1024];
    model->arch = new std::uint64_t[16 * 1024];  // max tensors allowed
    return model;
};

extern "C" void fairseq2_model_free(fairseq2_model* model) {
    delete (std::uint64_t*)(model->arch);
    delete (std::uint8_t*)model->hparams;
    delete model;
};

extern "C" std::string* std_string_alloc(char* c_str) {
    return new std::string(c_str);
}

extern "C" void std_string_free(std::string* str) {
    delete str;
}



// Linear

std::size_t Linear_size(int32_t input_dim, int32_t output_dim)
{
    return (input_dim * output_dim * ggml_type_size(GGML_TYPE_F32)) // weight
        + (output_dim * ggml_type_size(GGML_TYPE_F32)); // bias
};

void Linear_init(
    Linear& self,
    fairseq2_model& model,
    const std::string &prefix,
    int input_dim,
    int output_dim,
    bool bias
) {
    self.weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, output_dim, input_dim);
    model.tensors[prefix + ".weight"] = self.weight;
    if (bias) {
        self.bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, output_dim);
        model.tensors[prefix + ".inner_proj.bias"] = self.bias;
    }
}

extern "C" ggml_tensor*
Linear_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input  // (d_in)
) {
    // Note: for now we assumed un-batched input
    ggml_tensor* weight = model.tensors[prefix + ".weight"];  // (d_in, d_out)
    ggml_tensor* bias = model.tensors[prefix + ".bias"];  // (d_out)

    return ggml_add(
        model.ctx,
        ggml_mul_mat(model.ctx, weight, input),  // (d_out)
        bias
    );
}

// LayerNorm

std::size_t LayerNorm_size(int32_t dim)
{
    return 2 * dim * ggml_type_size(GGML_TYPE_F32); // weight and bias
};

void LayerNorm_init(
    LayerNorm& self,
    fairseq2_model& model,
    const std::string &prefix,
    int dim
) {
    self.weight = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, dim);
    model.tensors[prefix + ".weight"] = self.weight;
    self.bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, dim);
    model.tensors[prefix + ".bias"] = self.bias;
}

extern "C" ggml_tensor* LayerNorm_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input) {
    ggml_tensor* weight = model.tensors[prefix + ".weight"];
    ggml_tensor* bias = model.tensors[prefix + ".bias"];

    auto ctx = model.ctx;
    // TODO: should `eps` be part of unity hparams ?
    input = ggml_norm(ctx, input, /*eps*/1e-5);
    return ggml_add(
        ctx,
        ggml_mul(ctx, ggml_repeat(ctx, weight, input), input),
        ggml_repeat(ctx, bias, input)
    );
}


std::size_t StandardFeedForwardNetwork_size(int32_t dim, int32_t inner_dim)
{
    return LayerNorm_size(dim) + Linear_size(dim, inner_dim) + Linear_size(inner_dim, dim);
};

void StandardFeedForwardNetwork_init(
    StandardFeedForwardNetwork& self,
    fairseq2_model& model,
    const std::string &prefix,
    int model_dim,
    int inner_dim
) {
    Linear_init(self.inner_proj, model, prefix + ".inner_proj", model_dim, inner_dim, true);
    LayerNorm_init(self.inner_layer_norm, model, prefix + ".inner_layer_norm", inner_dim);
    Linear_init(self.output_proj, model, prefix + ".output_proj", inner_dim, model_dim, true);
}

extern "C" ggml_tensor* StandardFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    seqs = Linear_forward(model, prefix + ".inner_proj", seqs);
    // inner_activation = ReLu // TODO: allow other activation
    seqs = ggml_relu(model.ctx, seqs);

    if (model.tensors.find(prefix + ".inner_layer_norm.weight") != model.tensors.end()) {
        seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    }

    // TODO: inference dropout
    // if self.inner_dropout is not None:
    //     seqs = self.inner_dropout(seqs)
    seqs = Linear_forward(model, prefix + ".output_proj", seqs);
    return seqs;
}

void MultiheadAttention_init(
    MultiheadAttention& self,
    fairseq2_model& model,
    const std::string &prefix,
    int model_dim,
    int num_heads
) {
    int bias = true;
    int num_key_value_heads = num_heads;
    int head_dim = model_dim / num_heads;

    Linear_init(self.q_proj, model, prefix + ".q_proj", model_dim, model_dim, bias);
    Linear_init(self.k_proj, model, prefix + ".k_proj", model_dim, head_dim * num_key_value_heads, bias);
    Linear_init(self.v_proj, model, prefix + ".v_proj", model_dim, model_dim, bias);

    // (H, 1, K_h)
    self.bias_k = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, num_heads, 1, head_dim * num_key_value_heads/ num_heads);
    // (H, 1, V_h)
    self.bias_v = ggml_new_tensor_3d(model.ctx, GGML_TYPE_F32, num_heads, 1, model_dim / num_heads);
}

ggml_tensor* reshape_num_head(ggml_context* ctx, ggml_tensor* x, int num_heads) {
    int slen = x->ne[0];
    // (S, M) -> (S, K_proj)
    x = ggml_reshape_3d(ctx, x, slen, num_heads, x->ne[1] / num_heads);
    // (S, K_proj) -> (H, S, K_h)
    return ggml_transpose(ctx, x);
}



extern "C" ggml_tensor* // (d_in, seq_len)
MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (d_in, len_q)
    ggml_tensor* keys,  // (d_in, len_k)
    ggml_tensor* values,  // (d_out, len_k)
    ggml_tensor* mask // (seq_len, len_q)
) {
    int num_heads = 16;
    ggml_context* ctx = model.ctx;
    ggml_tensor* q = Linear_forward(model, prefix + ".q_proj", queries);
    q = reshape_num_head(ctx, q, num_heads);
    ggml_tensor* k = Linear_forward(model, prefix + ".k_proj", keys);
    k = reshape_num_head(ctx, k, num_heads);
    ggml_tensor* v = Linear_forward(model, prefix + ".q_proj", queries);
    v = reshape_num_head(ctx, v, num_heads);

    ggml_tensor* attn = ggml_flash_attn(model.ctx, q, k, v, /*masked*/true);
    attn = Linear_forward(model, prefix + ".output_proj", attn);
    return attn;
    // ggml_tensor* attn = SDPA_forward(q, k, v, nullptr);
    // // (H, S, V_h) -> (S, H, V_h)
    // attn = ggml_transpose(ctx, attn);
    // // (S, H, V_h) -> (S, V_proj)
    // attn = ggml_reshape_3d()
}

// extern "C" ggml_tensor* // (d_out, seq_len)
// SDPA_forward(
//     fairseq2_model& model,
//     const std::string &prefix,
//     ggml_tensor* queries,  // (d_in, len_q)
//     ggml_tensor* keys,  // (d_in, len_k)
//     ggml_tensor* values,  // (d_out, len_k)
//     ggml_tensor* mask // (seq_len, len_q)
// ) {
//     return queries;
// }
