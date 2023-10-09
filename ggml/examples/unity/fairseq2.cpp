#include <math.h>
#include "ggml.h"
#include "fairseq2.h"


/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc() {
    // pre-allocate some memory to write hyperparameters and tensors pointers
    auto* model = new fairseq2_model;
    model->hparams = new std::uint8_t[8 * 1024];
    model->arch = new std::uint64_t[16 * 1024];  // max tensors allowed
    model->tensors_ctx = nullptr;
    return model;
};

extern "C" void fairseq2_model_free(fairseq2_model* model) {
    if (model->tensors_ctx) ggml_free(model->tensors_ctx);
    delete (std::uint64_t*)(model->arch);
    delete (std::uint8_t*)model->hparams;
    delete model;
};

extern "C" void fairseq2_model_set_inference_ctx(fairseq2_model* model, ggml_context* ctx) {
    model->ctx = ctx;
}

extern "C" std::string* std_string_alloc(char* c_str) {
    return new std::string(c_str);
}

extern "C" void std_string_free(std::string* str) {
    delete str;
}


extern "C" ggml_tensor* Linear_forward(
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


ggml_tensor* reshape_num_head(ggml_context* ctx, ggml_tensor* x, int num_heads) {
    int slen = x->ne[1];
    int model_dim = x->ne[0];
    // (S, dim) -> (S, H, H_dim)
    x = ggml_reshape_3d(ctx, x, model_dim / num_heads, num_heads, slen);
    // (S, H, H_dim) -> (H, S, H_dim)
    x = ggml_permute(ctx, x, 0, 2, 1, 3);
    return x;
}


// TODO: borken
extern "C" ggml_tensor* MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (slen, d_in)
    ggml_tensor* keys,  // (klen, d_in)
    ggml_tensor* values,  // (klen, d_out)
    ggml_tensor* mask // (klen, slen)  TODO: do we need to pass mask here ?
) {
    int slen = queries->ne[1];
    int slenk = keys->ne[1];
    int num_heads = 16;
    int head_dim = queries->ne[0] / num_heads;
    ggml_context* ctx = model.ctx;
    ggml_tensor* q = Linear_forward(model, prefix + ".q_proj", queries);
    q = reshape_num_head(ctx, q, num_heads);  // (H, S, H_dim)
    ggml_set_name(q, "q");
    ggml_tensor* k = Linear_forward(model, prefix + ".k_proj", keys);
    k = reshape_num_head(ctx, k, num_heads);  // (H, Sk, H_dim)
    ggml_set_name(k, "k");

    ggml_tensor* v = Linear_forward(model, prefix + ".v_proj", values);
    v = ggml_reshape_3d(ctx, v, head_dim, num_heads, slenk); // (Sk, H, H_dim)
    v = ggml_permute(ctx, v, 1, 2, 0, 3);  // (H, H_dim, Sk)
    v = ggml_cont(ctx, v);
    ggml_set_name(v, "v");

    // (H, Sk, H_dim) x (H, S, H_dim) -> (H, S, Sk)
    ggml_tensor* qk = ggml_mul_mat(ctx, k, q);
    ggml_set_name(qk, "qk");
    ggml_tensor* qk_scale = ggml_new_tensor_1d(ctx, qk->type, 1);
    ggml_set_f32(qk_scale, 1.0f/sqrtf(float(head_dim)));
    qk = ggml_scale(ctx, qk, qk_scale);
    ggml_set_name(qk, "qk_scaled");

    if (mask) qk = ggml_add(ctx, qk, mask);
    // TODO: upgrade qk to float32 if needed
    ggml_tensor* attn_weights = ggml_soft_max(ctx, qk);  // (H, Sk, S)
    ggml_set_name(attn_weights, "attn_weights");

    // (H, S, Sk) x (H, H_dim, Sk) -> (H, H_dim, S)
    ggml_tensor* attn = ggml_mul_mat(ctx, attn_weights, v);
    ggml_set_name(attn, "attn");
    attn = ggml_reshape_2d(ctx, attn, slen, num_heads * head_dim); // (H * H_dim, S)
    attn = ggml_transpose(ctx, attn); // (S, H * H_dim)
    // // I'm not sure why this one is needed ...
    attn = ggml_cont(ctx, attn);
    // out -> (S, d_out)
    ggml_tensor* out = Linear_forward(model, prefix + ".output_proj", attn);
    ggml_set_name(out, "out");

    return out;
}

// ggml_tensor* attn_weights = ggml_mul_mat(ctx, q, k);  // (H, S, S)
//     attn_weights = ggm_mul * (q.size(-1) ** -0.5)

//     if mask is not None:
//         attn_weights = attn_weights + mask

//     # For numerical stability run in single precision.
//     attn_weights = softmax(attn_weights, dim=-1, dtype=torch.float32)

//     attn_weights = attn_weights.type_as(q)

//     if training and dropout_p > 0.0:
//         attn_weights = dropout(attn_weights, dropout_p)

//     # (*, S, S_kv) @ (*, S_kv, V) = (*, S, V)
//     attn = torch.matmul(attn_weights, values)

//     return attn, attn_weights if needs_weights else None

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
