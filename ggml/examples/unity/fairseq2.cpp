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
    GGML_ASSERT(weight != nullptr);
    ggml_tensor* bias = model.tensors[prefix + ".bias"];  // (d_out)
    GGML_ASSERT(bias != nullptr);

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
    GGML_ASSERT(weight != nullptr);
    ggml_tensor* bias = model.tensors[prefix + ".bias"];
    GGML_ASSERT(bias != nullptr);

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

# define UNITY_FLASH_ATTN

extern "C" ggml_tensor* MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (slen, d_in)
    ggml_tensor* keys,  // (klen, d_in)
    ggml_tensor* values,  // (klen, d_out)
    ggml_tensor* mask // (klen, slen)
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

#ifdef UNITY_FLASH_ATTN
    // For flash_attn, we assume either no masks, or triangular masks.
    ggml_tensor* attn = ggml_flash_attn(ctx, q, k, v, /*masked*/mask != nullptr);  // (H, S, H_dim)
    ggml_set_name(attn, "attn");
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3); // (S, H, H_dim)
    attn = ggml_cont(ctx, attn);
    attn = ggml_reshape_2d(ctx, attn, num_heads * head_dim, slen); // (S, H * H_dim)
#else
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
#endif  // UNITY_FLASH_ATTN
    // out -> (S, d_out)
    ggml_tensor* out = Linear_forward(model, prefix + ".output_proj", attn);
    ggml_set_name(out, "out");

    return out;
}


bool has_layer(fairseq2_model& model, const std::string& name) {
    return model.tensors.find(name) != model.tensors.end();
}


extern "C" ggml_tensor* StandardTransformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    // TODO: read norm_order from model
    auto norm_order = TRANSFORMER_NORM_ORDER_PRE;

    // _forward_self_attn(seqs, padding_mask)
    auto residual = seqs;
    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // TODO: add padding_mask to MultiheadAttention_forward
    GGML_ASSERT(padding_mask == nullptr);
    seqs = MultiheadAttention_forward(
        model,
        prefix + ".self_attn",
        seqs,
        seqs,
        seqs,
        /*attention masks=*/nullptr
    );

    if (has_layer(model, prefix + ".self_attn_norm.weight"))
        seqs = LayerNorm_forward(model, prefix + ".self_attn_norm", seqs);

    // TODO: seqs = self.self_attn_dropout(seqs)

    seqs = ggml_add(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // _forward_ffn(seqs)
    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);

    // TODO:
    // seqs = self.ffn_dropout(seqs)
    // if self.residual_scale is not None:
    // residual = self.residual_scale * residual

    seqs = ggml_add(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    return seqs;
}
