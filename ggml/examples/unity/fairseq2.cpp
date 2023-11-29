#include <algorithm>
#include <fnmatch.h>
#include <iostream>
#include <math.h>
#include <queue>
#include <unordered_map>

#include "kaldi-native-fbank/csrc/feature-fbank.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "fairseq2.h"
#include "ggml.h"

ggml_tensor* ggml_detach(ggml_tensor* a) {
    a->op = GGML_OP_NONE;
    std::fill(a->src, a->src + GGML_MAX_SRC, nullptr);
    return a;
}

#define DEBUG_MEM_USAGE 0

void printf_mem_usage(ggml_context* ctx, std::string name) {
#if DEBUG_MEM_USAGE
    double mb = 1024.0 * 1024.0;
    printf(
        "ctx %s: memory used = %8.2f MB, memory reserved = %8.2f Mb\n",
        name.c_str(),
        ggml_used_mem(ctx) / mb,
        ggml_get_mem_size(ctx) / mb
    );
#endif
}

#define SWAP(x, y) \
    auto tmp_ ## x = x; x = y; y = tmp_ ## x;


/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc() {
    // pre-allocate some memory to write hyperparameters and tensors pointers
    auto* model = new fairseq2_model;
    model->tensors_ctx = nullptr;
    return model;
}

extern "C" void fairseq2_kv_cache_alloc(const fairseq2_model& model, int beam_size, int max_seq_len) {
    // Note: we only allocate the cache for the decoder attention.
    // For encoder attention since we compute it all at once,
    // the allocation is delayed to the first forward pass, to not over allocate.
    auto attn_glob = "text_decoder.*_attn.k_proj.weight";
    auto self_attn_glob = "text_decoder.*self_attn.k_proj.weight";
    ggml_tensor* self_attn_mask = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, max_seq_len, max_seq_len);
    self_attn_mask = ggml_diag_mask_inf_inplace(model.ctx, self_attn_mask, 0);
    ggml_format_name(self_attn_mask, "self_attn_mask[%d]", max_seq_len);

    for (auto named_tensor : model.tensors) {
        const std::string& name = named_tensor.first;
        if (::fnmatch(attn_glob, name.c_str(), 0) == FNM_NOMATCH)
            continue;
        // create a cache entry without the ".k_proj.weight" suffix
        const std::string& shortname = name.substr(0, name.size() - 14);
        KeyValueTensor& kv = model.kv_cache[shortname];
        kv.step_nr = 0;

        if (::fnmatch(self_attn_glob, name.c_str(), 0) == FNM_NOMATCH) {
            // enc_dec_attn
            // the tensors will be allocated during the first forward
            continue;
        }

        // self_attn
        ggml_tensor* k_proj = named_tensor.second;
        int model_dim = k_proj->ne[0];
        kv.full_k = ggml_new_tensor_3d(model.ctx, k_proj->type, model_dim, max_seq_len, beam_size);
        kv.full_v = ggml_new_tensor_3d(model.ctx, k_proj->type, model_dim, max_seq_len, beam_size);
        kv.self_attn_mask = self_attn_mask;
        ggml_format_name(kv.full_k, "%s.k_cache", shortname.c_str());
        ggml_format_name(kv.full_v, "%s.v_cache", shortname.c_str());
    }
}

extern "C" void fairseq2_kv_cache_reset(const fairseq2_model& model) {
    // TODO: use a dedicated allocator, so that kv_cache.clear actually frees the memory
    model.kv_cache.clear();
}


bool has_kv_cache(const fairseq2_model& model) {
    return model.kv_cache.size() > 0;
}

// copy k and v to kv cache
// kv.full_k[step_nr] = k;
// kv.full_v[step_nr] = v;
void append_to_prev_kv(const fairseq2_model& model, const std::string& prefix, ggml_tensor** k, ggml_tensor** v, ggml_tensor** self_attn_mask) {
    KeyValueTensor& kv = model.kv_cache[prefix];
    GGML_ASSERT(kv.full_k != nullptr); // key not found !
    int step_nr = kv.step_nr;

    ggml_tensor* full_k = kv.full_k;
    ggml_tensor* full_v = kv.full_v;

    // (N, S_kv, K_proj)
    GGML_ASSERT((*k)->ne[1] == 1);  // TODO I think we could handle adding a full prefix sequence
    ggml_tensor* updated_k = ggml_set_2d_inplace(model.ctx, full_k, *k, full_k->nb[2], full_k->nb[1] * step_nr);
    ggml_tensor* updated_v = ggml_set_2d_inplace(model.ctx, full_v, *v, full_v->nb[2], full_v->nb[1] * step_nr);

    *k = ggml_slice(model.ctx, updated_k, 1, 0, step_nr + 1);
    *v = ggml_slice(model.ctx, updated_v, 1, 0, step_nr + 1);
    ggml_format_name(*k, "%s (step=%d)", full_k->name, step_nr);
    ggml_format_name(*v, "%s (step=%d)", full_v->name, step_nr);

    // qk is (B * H, Sq, Sk) == (B*H, 1, Sk) in incremental mode
    // we return the Sq slice of the (Sq, Sk) attention mask
    *self_attn_mask = ggml_slice(
        model.ctx,
        ggml_slice(model.ctx, kv.self_attn_mask, 0, 0, step_nr + 1),
        1,
        step_nr,
        step_nr + 1
    );

    kv.step_nr = step_nr + 1;
}

// variant of ggml_get_rows that allows for a with more than 2 dims.
ggml_tensor* ggml_get_rows2(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    int flattened = 0;
    GGML_ASSERT(a->n_dims <= 3);
    if (a->n_dims == 3) {
        flattened = a->ne[0];
        a = ggml_flatten_1d(ctx, a, 0);
    }
    a = ggml_get_rows(ctx, a, b);
    if (flattened) {
        a = ggml_unflatten_1d(ctx, a, 0, flattened);
    }
    return a;
}


void _reorder_kv_cache(ggml_context* ctx, ggml_cgraph* gf, KeyValueTensor& kv, ggml_tensor* new_order) {
    if (kv.full_k != nullptr) {
        ggml_detach(kv.full_k);
        kv.full_k = ggml_get_rows2(ctx, kv.full_k, new_order);
        ggml_build_forward_expand(gf, kv.full_k);
    }

    if (kv.full_v != nullptr) {
        ggml_detach(kv.full_v);
        kv.full_v = ggml_get_rows2(ctx, kv.full_v, new_order);
        ggml_build_forward_expand(gf, kv.full_v);
    }
}


void reorder_kv_cache(const fairseq2_model& model, ggml_context* ctx, ggml_cgraph* gf, ggml_tensor* new_order) {
    for (auto& named_kv : model.kv_cache) {
        _reorder_kv_cache(ctx, gf, named_kv.second, new_order);
    }
}


inline double model_layer_config_d(const fairseq2_model& model, std::string name) {
    const std::int64_t* data = &model.layer_config.at(name);
    double val = *(const double*)data;
    return val;
}

extern "C" double fairseq2_model_layer_config_double(const fairseq2_model& model, const char* name) {
    return model_layer_config_d(model, std::string(name));
}

extern "C" std::int64_t fairseq2_model_layer_config_int(const fairseq2_model& model, const char* name) {
    return model.layer_config.at(std::string(name));
}


extern "C" void fairseq2_model_free(fairseq2_model* model) {
    if (model->tensors_ctx) ggml_free(model->tensors_ctx);
    delete model;
}

extern "C" void fairseq2_model_set_inference_ctx(fairseq2_model* model, ggml_context* ctx) {
    model->ctx = ctx;
}

extern "C" std::string* std_string_alloc(char* c_str) {
    return new std::string(c_str);
}

extern "C" void std_string_free(std::string* str) {
    delete str;
}

bool has_layer(fairseq2_model& model, const std::string& name) {
    return model.tensors.find(name) != model.tensors.end();
}

ggml_tensor* mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b) {
    if (b->ne[1] == 1 && b->ne[2] > 1 &&  a->n_dims == 2) {
        // `b` has shape (B, 1, D).
        // if `a` is (D_out, D), then we do one matmul for the full batch.
        b = ggml_flatten_1d(ctx, b, 1);
        return ggml_unflatten_1d(ctx, ggml_mul_mat(ctx, a, b), 1, 1);
    }
    // there is also the k * q matmul -> (D, 1, B) * (D, 1, B) -> (1, 1, B)
    // not sure what's the best way to compute this with BLAS

    return ggml_mul_mat(ctx, a, b);  // (d_out)
}


extern "C" ggml_tensor* Linear_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input  // (d_in)
) {
    // Note: for now we assumed un-batched input
    ggml_tensor* weight = model.tensors[prefix + ".weight"];  // (d_in, d_out)
    GGML_ASSERT(weight != nullptr);
    ggml_tensor* out = mul_mat(model.ctx, weight, input);  // (d_out)
    ggml_tensor* bias = model.tensors[prefix + ".bias"];  // (d_out)
    if (bias == nullptr) return out;

    return ggml_add_inplace(model.ctx, out, bias);
}

extern "C" ggml_tensor* LayerNorm_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input
) {
    ggml_tensor* weight = model.tensors[prefix + ".weight"];
    GGML_ASSERT(weight != nullptr);
    ggml_tensor* bias = model.tensors[prefix + ".bias"];
    GGML_ASSERT(bias != nullptr);

    auto ctx = model.ctx;
    double eps = model_layer_config_d(model, prefix + ".eps");

    input = ggml_norm(ctx, input, /*eps*/eps);
    return ggml_add_inplace(
        ctx,
        ggml_mul_inplace(ctx, ggml_repeat(ctx, weight, input), input),
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
    seqs = ggml_relu_inplace(model.ctx, seqs);

    if (has_layer(model, prefix + ".inner_layer_norm")) {
        seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    }

    seqs = Linear_forward(model, prefix + ".output_proj", seqs);
    return seqs;
}

extern "C" ggml_tensor* SiluFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    seqs = Linear_forward(model, prefix + ".inner_proj", seqs);
    seqs = ggml_silu(model.ctx, seqs);

    if (has_layer(model, prefix + ".inner_layer_norm")) {
        seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    }

    seqs = Linear_forward(model, prefix + ".output_proj", seqs);
    return seqs;
}

ggml_tensor* ggml_flatten_1d(ggml_context* ctx, ggml_tensor* x, int dim) {
    int n_dims = x->n_dims;
    GGML_ASSERT(dim >= 0);
    GGML_ASSERT(dim < n_dims);
    GGML_ASSERT(ggml_is_contiguous(x));
    // Nothing to do
    if (dim == n_dims - 1) return x;

    if (n_dims == 2) {
        return ggml_reshape_1d(ctx, x, x->ne[0] * x->ne[1]);
    } else if (n_dims == 3) {
        if (dim == 0) {
            return ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]);
        } else { // dim == 1
            return ggml_reshape_2d(ctx, x, x->ne[0], x->ne[1] * x->ne[2]);
        }
    } else { // n_dims == 4
        if (dim == 0) {
            return ggml_reshape_3d(ctx, x, x->ne[0] * x->ne[1], x->ne[2], x->ne[3]);
        } else if (dim == 1) {
            return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1] * x->ne[2], x->ne[3]);
        } else { // dim == 2
            return ggml_reshape_3d(ctx, x, x->ne[0], x->ne[1], x->ne[2] * x->ne[3]);
        }
    }
}

ggml_tensor* ggml_unflatten_1d(ggml_context* ctx, ggml_tensor* x, int dim, int num_el) {
    int n_dims = x->n_dims;
    GGML_ASSERT(dim >= 0);
    GGML_ASSERT(dim < n_dims);
    GGML_ASSERT(n_dims < 4);
    GGML_ASSERT(x->ne[dim] % num_el == 0);
    GGML_ASSERT(x->nb[dim + 1] == x->nb[dim] * x->ne[dim]);  // `x` isn't contiguous along `dim`
    if (n_dims == 1) {
        return ggml_view_2d(ctx, x, num_el, x->ne[0] / num_el, x->nb[0] * num_el, 0);
    } else if (n_dims == 2) {
        if (dim == 0) {
            return ggml_view_3d(ctx, x, num_el, x->ne[0] / num_el, x->ne[1], x->nb[0] * num_el, x->nb[1], 0);
        } else { // dim == 1
            return ggml_view_3d(ctx, x, x->ne[0], num_el, x->ne[1] / num_el, x->nb[1], num_el * x->nb[1], 0);
        }
    } else { // (n_dims == 3)
        if (dim == 0) {
            return ggml_view_4d(ctx, x, num_el, x->ne[0] / num_el, x->ne[1], x->ne[2], x->nb[0] * num_el, x->nb[1], x->nb[2], 0);
        } else if (dim == 1) {
            return ggml_view_4d(ctx, x, x->ne[0], num_el, x->ne[1] / num_el, x->ne[2], x->nb[1], num_el * x->nb[1], x->nb[2], 0);
        } else { // dim == 2
            return ggml_view_4d(ctx, x, x->ne[0], x->ne[1], num_el, x->ne[2] / num_el, x->nb[1], x->nb[2], num_el * x->nb[2], 0);
        }
    }
}


ggml_tensor* _reshape_num_head(ggml_context* ctx, ggml_tensor* x, int head_dim) {
    // (B, S, dim) -> (B, S, H, H_dim)
    x = ggml_unflatten_1d(ctx, x, 0, head_dim);
    x = ggml_permute(ctx, x, 0, 2, 1, 3); // (B, H, S, H_dim)
    x = ggml_cont(ctx, x);
    x = ggml_flatten_1d(ctx, x, 2);  // (B * H, S, H_dim)
    return x;
}

/// (B, Sk, dim) -> // (B?, H, H_dim, Sk)
ggml_tensor* _reshape_num_head_values(ggml_context* ctx, ggml_tensor* v, int head_dim ) {
    // (B, Sk, dim) -> (B, Sk, H, H_dim)
    v = ggml_unflatten_1d(ctx, v, 0, head_dim);
    v = ggml_permute(ctx, v, 1, 2, 0, 3);  // (B?, H, H_dim, Sk)
    v = ggml_cont(ctx, v);
    v = ggml_flatten_1d(ctx, v, 2);  // (B * H, S, H_dim)
    return v;
}


// flash_attn doesn't work for cross attention because it assumes Q <= K
// and it seems to yield slightly different scores than expected, and thus a different beam search
# define UNITY_FLASH_ATTN 0

extern "C" ggml_tensor* MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (slen, d_in)
    ggml_tensor* keys,  // (klen, d_in)
    ggml_tensor* values,  // (klen, d_out)
    ggml_tensor* attn_mask // (klen, slen)
) {
    int model_dim = queries->ne[0];
    int num_heads = model.layer_config.at(prefix + ".num_heads");
    int head_dim = model_dim / num_heads;
    GGML_ASSERT(model_dim % num_heads == 0);

    ggml_context* ctx = model.ctx;
    ggml_tensor* q = Linear_forward(model, prefix + ".q_proj", queries); // (B, S, H * H_dim)
    ggml_set_name(q, "q");
    q = _reshape_num_head(ctx, q, head_dim);  // (B * H, S, H_dim)

    ggml_tensor *k, *v;
    if (!has_kv_cache(model)) {
        k = Linear_forward(model, prefix + ".k_proj", keys);
        ggml_set_name(k, "k");
        v = Linear_forward(model, prefix + ".v_proj", values);
        ggml_set_name(v, "v");
    } else {
        bool encoder_decoder_attn = keys == values && keys != queries;
        if (encoder_decoder_attn) {
            // The K and V tensors of an encoder-decoder attention (i.e. the
            // projected encoder outputs) remain static during evaluation.

            KeyValueTensor& kv_cache = model.kv_cache[prefix];
            if (kv_cache.step_nr == 0) {
                k = Linear_forward(model, prefix + ".k_proj", keys);
                v = Linear_forward(model, prefix + ".v_proj", values);
                // TODO: encoder_padding_mask
                // Note we are only storing a pointer to the buffer, not the full graph
                kv_cache.full_k = ggml_detach(ggml_dup_inplace(ctx, k));
                ggml_format_name(kv_cache.full_k, "%s.k_cache", prefix.c_str());
                kv_cache.full_v = ggml_detach(ggml_dup_inplace(ctx, v));
                ggml_format_name(kv_cache.full_v, "%s.v_cache", prefix.c_str());
                kv_cache.step_nr = keys->ne[1];
            } else {
                k = kv_cache.full_k;
                v = kv_cache.full_v;
                // This is a cache collision. TODO: fairseq2_kv_cache_reset
                GGML_ASSERT(keys->ne[1] == k->ne[1]);
                GGML_ASSERT(values->ne[1] == v->ne[1]);
            }
        } else { // self attention
            // (1, K) -> (N, 1, K_proj)
            k = Linear_forward(model, prefix + ".k_proj", keys);
            ggml_set_name(k, "k");
            // (1, V) -> (N, 1, V_proj)
            v = Linear_forward(model, prefix + ".v_proj", values);
            ggml_set_name(v, "v");

            append_to_prev_kv(model, prefix, &k, &v, &attn_mask);
        }
    }
    k = _reshape_num_head(ctx, k, head_dim);  // (B * H, Sk, H_dim)
    v = _reshape_num_head_values(ctx, v, head_dim); // (B * H, H_dim, Sk)
    v = ggml_cont(ctx, v);

#if UNITY_FLASH_ATTN
    // For flash_attn, we assume either no masks, or triangular masks.
    ggml_tensor* attn = ggml_flash_attn(ctx, q, k, v, /*masked*/attn_mask != nullptr);  // (B * H, S, H_dim)
    ggml_set_name(attn, "attn");
    attn = ggml_unflatten_1d(ctx, attn, 2, num_heads);  // (B, H, H_dim, S)
    attn = ggml_permute(ctx, attn, 0, 2, 1, 3); // (B, S, H, H_dim)
#else
    // (B * H, Sk, H_dim) x (B * H, S, H_dim) -> (B * H, S, Sk)
    ggml_tensor* qk = mul_mat(ctx, k, q);
    ggml_set_name(qk, "qk");
    ggml_tensor* qk_scale = ggml_new_tensor_1d(ctx, qk->type, 1);
    ggml_set_f32(qk_scale, 1.0f/sqrtf(float(head_dim)));
    qk = ggml_scale_inplace(ctx, qk, qk_scale);
    ggml_set_name(qk, "qk_scaled");

    // TODO: Should we replace this by ggml_diag_mask_inf ?
    if (attn_mask) qk = ggml_add_inplace(ctx, qk, attn_mask);
    // TODO: upgrade qk to float32 if needed
    ggml_tensor* attn_weights = ggml_soft_max(ctx, qk);  // (B * H, S, Sk)
    ggml_set_name(attn_weights, "attn_weights");

    // (B * H, S, Sk) x (B * H, H_dim, Sk) -> (B * H, H_dim, S)
    ggml_tensor* attn = mul_mat(ctx, attn_weights, v);
    ggml_set_name(attn, "attn");
    attn = ggml_unflatten_1d(ctx, attn, 2, num_heads);  // (B, H, H_dim, S)
    attn = ggml_permute(ctx, attn, 2, 0, 1, 3); // (B, S, H, H_dim)
#endif  // UNITY_FLASH_ATTN
    attn = ggml_cont(ctx, attn);
    attn = ggml_flatten_1d(ctx, attn, 0); // (B, S, H * H_dim)
    // out -> (B, S, d_out)
    ggml_tensor* out = Linear_forward(model, prefix + ".output_proj", attn);
    ggml_set_name(out, "out");

    return out;
}


extern "C" ggml_tensor* StandardTransformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    auto norm_order = model.layer_config.at(prefix + ".norm_order");

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
        /*attn_mask=*/nullptr
    );

    if (has_layer(model, prefix + ".self_attn_norm"))
        seqs = LayerNorm_forward(model, prefix + ".self_attn_norm", seqs);

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // _forward_ffn(seqs)
    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);

    // TODO: if self.residual_scale is not None:
    // residual = self.residual_scale * residual

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* WaveformToFbank_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* waveform
) {
    // Hardcoding: num_bins 80, sample rate 16k, always standardize
    ggml_context* ctx = model.ctx;
    knf::MelBanksOptions mel_opts{};
    mel_opts.num_bins = 80;

    knf::FrameExtractionOptions frame_opts{};
    frame_opts.samp_freq = 16000;

    knf::FbankOptions opts{};
    opts.frame_opts = frame_opts;
    opts.mel_opts = mel_opts;


    std::vector<float_t> signal_frame{};
    std::int32_t num_frames = knf::NumFrames(/*num_samples=*/waveform->ne[0], frame_opts);
    ggml_tensor* output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 80, num_frames);
    knf::FbankComputer native_(opts);
    knf::FeatureWindowFunction window_fn_(native_.GetFrameOptions());

    for (std::int32_t frame_nr = 0; frame_nr < num_frames; ++frame_nr) {
        signal_frame.resize(0);

        // Extract the frame from the waveform tensor.
        knf::ExtractWindow(
            /*sample_offset=*/0,
            (float *)(waveform->data),
            waveform->ne[0],
            frame_nr,
            frame_opts,
            window_fn_,
            &signal_frame);

        native_.Compute(
            /*signal_raw_log_energy=*/0, /*vtln_warp=*/1.0, &signal_frame, ((float *)(output->data) + frame_nr * 80));
    }
    output = ggml_dup(ctx, ggml_transpose(ctx, output));
    output = ggml_norm(ctx, output, 1e-5);
    output = ggml_dup(ctx, ggml_transpose(ctx, output));
    if (output->ne[1] % 2 == 1) {
        ggml_tensor* remove_last = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, output->ne[1]-1);
        for (int i = 0; i < output->ne[1]-1; ++i) {
            ((int32_t *) remove_last->data)[i] = i;
        }
        output = ggml_get_rows(ctx, output, remove_last);
    }
    output = ggml_reshape_2d(ctx, output, output->ne[0] * 2, output->ne[1] / 2);
    return output;
}

// TODO: Check if it's possible to merge with standard MHA
extern "C" ggml_tensor* RelativePositionMHA_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    ggml_context* ctx = model.ctx;

    ggml_tensor* residual = seqs;
    seqs = LayerNorm_forward(model, prefix + "_layer_norm", seqs);
    // self_attn: qkv
    ggml_tensor* Qcur = Linear_forward(model, prefix + ".q_proj", seqs);
    ggml_tensor* Kcur = Linear_forward(model, prefix + ".k_proj", seqs);
    ggml_tensor* Vcur = Linear_forward(model, prefix + ".v_proj", seqs);

    // self_attn: rel_pos SDPA
    int32_t S = seqs->ne[1];
    int32_t H = 16; // TODO: Make this configurable
    int32_t n_ctx = 4096;
    int32_t K_h = seqs->ne[0] / H;

    int32_t start_index = n_ctx - S;
    int32_t end_index = n_ctx + S - 1;

    int num_indices = end_index - start_index;

    ggml_tensor* rows = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, num_indices);
    for (int i = 0; i < num_indices; i++) {
        ((int32_t *)rows->data)[i] = start_index + i;
    }

    // self_attn: load pos_enc weights & compute_r
    // In fairseq2 pos_enc weights are calculated on the fly, since some more custom operators might be needed to enable this,
    // we store the results (fixed) in checkpoint as model.audio_enc_pos_enc_w and load directly.
    ggml_tensor* r = ggml_get_rows(ctx, model.tensors["speech_encoder.pos_enc"], rows);
    r = mul_mat(ctx, model.tensors[prefix + ".sdpa.r_proj.weight"], r);
    r = ggml_dup(ctx, ggml_permute(ctx,
                        ggml_cpy(ctx,
                            r,
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K_h, H, S*2-1)),
                        0, 2, 1, 3));

    ggml_tensor* u_bias = ggml_reshape_3d(ctx, model.tensors[prefix + ".sdpa.u_bias"], K_h, 1, H);
    ggml_tensor* v_bias = ggml_reshape_3d(ctx, model.tensors[prefix + ".sdpa.v_bias"], K_h, 1, H);

    // self_attn: Permute QKV

    ggml_tensor* Q = ggml_cont(ctx, ggml_permute(ctx,
                        ggml_cpy(ctx,
                            Qcur,
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K_h, H, S)),
                        0, 2, 1, 3)); // (H * K_h, S) -> (K_h, H, S) -> (K_h, S, H)
    ggml_tensor* K = ggml_cont(ctx, ggml_permute(ctx,
                        ggml_cpy(ctx,
                            Kcur,
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K_h, H, S)),
                        0, 2, 1, 3)); // (H * K_h, S) -> (K_h, H, S) -> (K_h, S, H)
    ggml_tensor* V = ggml_cont(ctx, ggml_permute(ctx,
                        ggml_cpy(ctx,
                            Vcur,
                            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K_h, H, S)),
                        1, 2, 0, 3)); // (H * K_h, S) -> (K_h, H, S) -> (H, S, K_h)


    ggml_tensor* q_with_u_bias = ggml_add_inplace(ctx, ggml_dup(ctx, Q), u_bias); // (K_h, S, H)
    ggml_tensor* q_with_v_bias = ggml_add_inplace(ctx, Q, v_bias); // (K_h, S, H)

    ggml_tensor* ac = mul_mat(ctx, K, q_with_u_bias);
    ggml_tensor* bd = mul_mat(ctx, r, q_with_v_bias);


    // self_attn: shift_bd. Logic follows https://github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/nn/transformer/relative_attention.py#L161
    bd = ggml_dup(ctx, ggml_permute(ctx, bd, 2, 1, 0, 3)); // H, S, 2S-1

    ggml_tensor* pad = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, H, S, 1);
    pad = ggml_set_f32(pad, 0.0);

    bd = ggml_concat(ctx, pad, bd); // bd[i][j][0] == 0, (H, S, 2S)
    bd = ggml_dup(ctx, ggml_permute(ctx, bd, 2, 1, 0, 3)); // (2S, S, H)
    bd = ggml_reshape_3d(ctx, bd, S, 2 * S, H);  // (S, 2S, H)
    // discard the first set of positive positions
    bd = ggml_dup(ctx, ggml_slice(ctx, bd, 1, 1, 2 * S));
    // shifts each row by an extra step
    bd = ggml_reshape_3d(ctx, bd, 2 * S - 1, S, H);
    // Discard positions used for shift.
    bd = ggml_slice(ctx, bd, 0, 0, S);

    // self_attn: compute attn / weights
    ggml_tensor* attn_weights = ggml_add_inplace(ctx, ac, bd);
    ggml_tensor* attn_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_f32(attn_scale, 1.0 / pow(K_h, 0.5));
    attn_weights = ggml_mul_inplace(ctx, attn_weights, ggml_repeat(ctx, attn_scale, attn_weights));
    attn_weights = ggml_soft_max(ctx, attn_weights);

    ggml_tensor* attn = mul_mat(ctx, V, attn_weights); // K_h, S, H
    attn = ggml_dup(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    ggml_tensor* attn_2d = ggml_reshape_2d(ctx, attn, K_h * H, S);

    ggml_tensor* attn_out = mul_mat(ctx, model.tensors[prefix + ".output_proj.weight"], attn_2d);
    attn_out = ggml_add_inplace(
        ctx,
        attn_out,
        ggml_repeat(ctx, model.tensors[prefix + ".output_proj.bias"], attn_out)
    );
    attn_out = ggml_add_inplace(ctx, attn_out, residual);
    return attn_out;
}

extern "C" ggml_tensor* ConvModule_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
        ggml_context* ctx = model.ctx;
        ggml_tensor* residual = seqs;
        seqs = LayerNorm_forward(model, prefix + "_layer_norm", seqs);
        // conv: Use matmul for pointwise conv 1 - kernel_size=1, no padding case
        seqs = mul_mat(ctx, model.tensors[prefix + ".pointwise_conv1.weight"], seqs);

        // conv: GLU
        seqs = ggml_glu(ctx, seqs);
        seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));

        // S x C -> (S+K-1) x C -> K x S x C -> S x C
        seqs = ggml_conv_1d(ctx, model.tensors[prefix + ".depthwise_conv.weight"], seqs, 1, 15, 1);

        // conv: Custom implementation of batch norm
        seqs = ggml_batch_norm(ctx, seqs, model.tensors[prefix + ".batch_norm.weight"], model.tensors[prefix + ".batch_norm.bias"], model.tensors[prefix + ".batch_norm.running_mean"], model.tensors[prefix + ".batch_norm.running_var"], 1e-5);

        // conv: SiLU actvation
        seqs = ggml_silu_inplace(ctx, seqs);
        seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));

        // conv: Use matmul for pointwise conv 2 - kernel_size=1, no padding case
        seqs = mul_mat(ctx, model.tensors[prefix + ".pointwise_conv2.weight"], seqs);

        // conv: + residual
        seqs = ggml_add_inplace(ctx, seqs, residual);
        return seqs;
}

extern "C" ggml_tensor* StandardConformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    ggml_tensor* ffn_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_f32(ffn_scale, 0.5f);
    ggml_tensor* residual = seqs;
    seqs = LayerNorm_forward(model, prefix + ".ffn1_layer_norm", seqs);
    seqs = SiluFeedForwardNetwork_forward(model, prefix + ".ffn1", seqs);
    seqs = ggml_mul_inplace(ctx, seqs, ggml_repeat(ctx, ffn_scale, seqs));
    seqs = ggml_add_inplace(ctx, seqs, residual);
    seqs = RelativePositionMHA_forward(model, prefix + ".self_attn", seqs);
    seqs = ConvModule_forward(model, prefix + ".conv", seqs);
    residual = seqs;
    seqs = LayerNorm_forward(model, prefix + ".ffn2_layer_norm", seqs);
    seqs = SiluFeedForwardNetwork_forward(model, prefix + ".ffn2", seqs);
    seqs = ggml_mul_inplace(ctx, seqs, ggml_repeat(ctx, ffn_scale, seqs));
    seqs = ggml_add_inplace(ctx, seqs, residual);
    seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);
    return seqs;
}

extern "C" ggml_tensor* StandardConformerEncoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    seqs = WaveformToFbank_forward(model, prefix, seqs);
    seqs = LayerNorm_forward(model, prefix + "_frontend.post_extract_layer_norm", seqs);
    seqs = Linear_forward(model, prefix + "_frontend.model_dim_proj", seqs);
    int layer_idx = 0;

    std::string layer_name = prefix + ".inner.layers." + std::to_string(layer_idx);

    while (has_layer(model, layer_name)) {
        seqs = StandardConformerEncoderLayer_forward(
            model, layer_name, seqs, padding_mask
        );
        ggml_set_name(seqs, ("x_enc_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".inner.layers." + std::to_string(layer_idx);
    }

    seqs = LayerNorm_forward(model, prefix + ".inner_layer_norm", seqs);
    ggml_tensor* residual = seqs;
    seqs = Linear_forward(model, prefix + ".proj1", seqs);
    seqs = ggml_relu_inplace(ctx, seqs);
    seqs = Linear_forward(model, prefix + ".proj2", seqs);
    ggml_tensor* ffn_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 1);
    ggml_set_f32(ffn_scale, 0.5f);
    seqs = ggml_mul(ctx, ggml_repeat(ctx, ffn_scale, seqs), seqs);
    seqs = ggml_add_inplace(ctx, seqs, residual);
    layer_idx = 0;
    layer_name = prefix + ".adaptor_layers." + std::to_string(layer_idx);
    while (has_layer(model, layer_name)) {
        seqs = StandardConformerEncoderAdaptorLayer_forward(
            model, layer_name, seqs, padding_mask
        );
        ggml_set_name(seqs, ("x_ada_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".adaptor_layers." + std::to_string(layer_idx);
    }
    seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* StandardConformerEncoderAdaptorLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    ggml_context* ctx = model.ctx;
    ggml_tensor* residual = seqs;
    residual = LayerNorm_forward(model, prefix + ".residual_layer_norm", residual);
    residual = ggml_dup(ctx, ggml_permute(ctx, residual, 1, 0, 2, 3));
    residual = ggml_conv_1d_generic(ctx, model.tensors[prefix + ".residual_conv.weight"], residual, 8, 4, 1);
    residual = ggml_dup(ctx, ggml_permute(ctx, residual, 1, 0, 2, 3));
    residual = ggml_add_inplace(ctx, ggml_repeat(ctx, model.tensors[prefix + ".residual_conv.bias"], residual), residual);
    residual = ggml_glu(ctx, residual);

    seqs = LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);
    seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));
    seqs = ggml_conv_1d_generic(ctx, model.tensors[prefix + ".self_attn_conv.weight"], seqs, 8, 4, 1);
    seqs = ggml_dup(ctx, ggml_permute(ctx, seqs, 1, 0, 2, 3));
    seqs = ggml_add_inplace(ctx, seqs, ggml_repeat(ctx, model.tensors[prefix + ".self_attn_conv.bias"], seqs));
    seqs = ggml_glu(ctx, seqs);

    seqs = MultiheadAttention_forward(
        model,
        prefix + ".self_attn",
        seqs,
        seqs,
        seqs,
        /*attention masks=*/nullptr
    );
    seqs = ggml_add_inplace(ctx, seqs, residual);
    residual = seqs;
    seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);
    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);
    seqs = ggml_add_inplace(ctx, seqs, residual);
    return seqs;
}


/// ggml_slice(X, -1, start, end) is equivalent to X[start:end]
/// ggml_slice(X, 0, start, end) is equivalent to X[..., start:end]
ggml_tensor* ggml_slice(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    int axis,
    int64_t start,
    int64_t end
) {
    int64_t ne[4];
    std::copy(a->ne, a->ne + 4, ne);
    if (axis < 0) axis = a->n_dims + axis;
    if (start < 0) start = ne[axis] + start;
    if (end <= 0) end = ne[axis] + end;
    GGML_ASSERT(0 <= start);
    GGML_ASSERT(start < end);
    GGML_ASSERT(end <= ne[axis]);


    ne[axis] = end - start;
    size_t offset = a->nb[axis] * start;

    size_t* nb = a->nb;
    ggml_tensor* result = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], nb[1], nb[2], nb[3], offset);
    ggml_format_name(result, "%s [(%d)%ld:%ld]", a->name, axis, start, end);
    result->n_dims = a->n_dims;
    return result;
}

ggml_tensor* ggml_select(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    int axis,
    int64_t index
) {
    int64_t ne[GGML_MAX_DIMS];
    std::copy(a->ne, a->ne + GGML_MAX_DIMS, ne);

    if (axis < 0) axis = a->n_dims + axis;
    if (index < 0) index = ne[axis] + index;
    GGML_ASSERT(0 <= index);
    GGML_ASSERT(index < ne[axis]);

    std::copy(a->ne + axis + 1, a->ne + GGML_MAX_DIMS, ne + axis);

    size_t offset = a->nb[axis] * index;
    size_t* nb = a->nb;
    GGML_ASSERT(GGML_MAX_DIMS == 4);
    ggml_tensor* result = ggml_view_3d(ctx, a, ne[0], ne[1], ne[2], nb[1], nb[2], offset);
    ggml_format_name(result, "%s [(%d)%ld]", a->name, axis, index);
    result->n_dims = a->n_dims - 1;
    return result;
}


// Inplace computation of PositionalEmbedding
extern "C" ggml_tensor* PositionalEmbedding_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* embeds
) {
    // This only work with the simple pos encoders
    int seq_len = embeds->ne[1];
    ggml_tensor* full_pos_embeds = model.tensors[prefix];

    int start_step = 0;
    if (has_kv_cache(model)) {
        start_step = model.kv_cache[prefix].step_nr++;
    }
    ggml_tensor* pos_embeds = ggml_slice(model.ctx, full_pos_embeds, /*axis*/1, start_step, seq_len + start_step);
    return ggml_add(model.ctx, embeds, pos_embeds);
}

extern "C" ggml_tensor* TransformerEmbeddingFrontend_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
) {
    GGML_ASSERT(seqs->n_dims < GGML_MAX_DIMS);
    ggml_context* ctx = model.ctx;
    ggml_tensor* embed_weights = model.tensors[prefix + ".embed.weight"];
    GGML_ASSERT(embed_weights != nullptr);
    ggml_tensor* embeds;
    if (seqs->n_dims == 1) {
        embeds = ggml_get_rows(ctx, embed_weights, seqs);
    } else {
        // ggml_get_rows isn't very flexible, we have to handle the reshape ourselves.
        ggml_tensor* flat_seqs = seqs;
        if (!ggml_is_contiguous(seqs)) {
            flat_seqs->type = GGML_TYPE_F32;
            flat_seqs = ggml_cont(ctx, flat_seqs);
        }
        flat_seqs = ggml_reshape_1d(ctx, flat_seqs, ggml_nelements(seqs));
        flat_seqs->type = GGML_TYPE_I32;
        embeds = ggml_get_rows(ctx, embed_weights, flat_seqs);
        embeds = ggml_reshape_4d(ctx, embeds, embed_weights->ne[0], seqs->ne[0], seqs->ne[1], seqs->ne[2]);
        embeds->n_dims = seqs->n_dims + 1;
    }

    // padding mask ?
    // padding_mask = to_padding_mask(embeds, seq_lens)

    if (has_layer(model, prefix + ".pos_encoder")) {
        embeds = PositionalEmbedding_forward(model, prefix + ".pos_encoder", embeds);
    }

    if (has_layer(model, prefix + ".layer_norm")) {
        embeds = LayerNorm_forward(model, prefix + ".layer_norm", embeds);
    }

    return embeds;
}

extern "C" ggml_tensor* StandardTransformerEncoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
) {
    int layer_idx = 0;
    std::string layer_name = prefix + ".layers." + std::to_string(layer_idx);
    while (has_layer(model, layer_name)) {
        seqs = StandardTransformerEncoderLayer_forward(
            model, layer_name, seqs, padding_mask
        );

        ggml_set_name(seqs, ("x_enc_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".layers." + std::to_string(layer_idx);
    }

    if (has_layer(model, prefix + ".layer_norm"))
        seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* StandardTransformerDecoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* self_attn_mask,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask
) {
    ggml_context* ctx = model.ctx;
    auto norm_order = model.layer_config.at(prefix + ".norm_order");

    // _forward_self_attn(seqs, padding_mask)
    auto residual = seqs;
    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    seqs = MultiheadAttention_forward(
        model,
        prefix + ".self_attn",
        seqs,
        seqs,
        seqs,
        /*attn_mask=*/self_attn_mask
    );

    if (has_layer(model, prefix + ".self_attn_norm"))
        seqs = LayerNorm_forward(model, prefix + ".self_attn_norm", seqs);

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".self_attn_layer_norm", seqs);

    // _forward_encoder_decoder_attn
    if (! has_layer(model, prefix + ".encoder_decoder_attn")) {
        // `encoder_output` must be `None` for decoder-only attention.
        GGML_ASSERT(encoder_output == nullptr);
        return seqs;
    }

    // `encoder_output` must not be `None` for encoder-decoder attention.
    GGML_ASSERT(encoder_output != nullptr);

    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".encoder_decoder_attn_layer_norm", seqs);


    seqs = MultiheadAttention_forward(
        model,
        prefix + ".encoder_decoder_attn",
        seqs,
        encoder_output,
        encoder_output,
        /*attention masks=*/encoder_padding_mask
    );

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs =  LayerNorm_forward(model, prefix + ".encoder_decoder_attn_layer_norm", seqs);

    // _forward_ffn(seqs)
    residual = seqs;

    if (norm_order != TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    seqs = StandardFeedForwardNetwork_forward(model, prefix + ".ffn", seqs);

    // TODO:
    // if self.residual_scale is not None:
    // residual = self.residual_scale * residual

    seqs = ggml_add_inplace(ctx, seqs, residual);

    if (norm_order == TRANSFORMER_NORM_ORDER_POST)
        seqs = LayerNorm_forward(model, prefix + ".ffn_layer_norm", seqs);

    return seqs;
}

extern "C" ggml_tensor* causal_attention_mask(ggml_context* ctx, ggml_tensor* seqs) {
    auto seq_len = seqs->ne[1];
    // TODO: allow other ggml_type
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
    return ggml_diag_mask_inf(ctx, mask, 0);
}

extern "C" ggml_tensor* StandardTransformerDecoder_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask
) {
    int layer_idx = 0;
    std::string layer_name = prefix + ".layers." + std::to_string(layer_idx);
    ggml_tensor* self_attn_mask = causal_attention_mask(model.ctx, seqs);
    while (has_layer(model, layer_name)) {
        seqs = StandardTransformerDecoderLayer_forward(
            model, layer_name, seqs, self_attn_mask, encoder_output, encoder_padding_mask
        );

        ggml_set_name(seqs, ("x_dec_" + std::to_string(layer_idx)).c_str());
        layer_idx += 1;
        layer_name = prefix + ".layers." + std::to_string(layer_idx);
    }

    if (has_layer(model, prefix + ".layer_norm"))
        seqs = LayerNorm_forward(model, prefix + ".layer_norm", seqs);

    return seqs;
}


int _determine_max_seq_len(const SequenceGeneratorJob& job, int source_seq_len) {
    auto opts = job.opts;
    int max_seq_len = -1;
    if (source_seq_len <= 0 || opts.soft_max_seq_len_a <= 0) {
        max_seq_len = opts.hard_max_seq_len;
    } else {
        max_seq_len = std::min(opts.hard_max_seq_len, int(opts.soft_max_seq_len_a * source_seq_len) + opts.soft_max_seq_len_b);
    }

    if (opts.min_seq_len > max_seq_len) {
        printf(
            "The effective maximum sequence length must be greater than or equal to `min_seq_len` (%d), but is %d instead. Adjust your soft and hard maximum sequence length limits.\n",
            opts.min_seq_len,
            max_seq_len
        );
        GGML_ASSERT(opts.min_seq_len <= max_seq_len);
    }

    int prefix_seq_len = job.prefix_seq->ne[0];
    if (prefix_seq_len >= max_seq_len) {
        printf(
            "The effective maximum sequence length must be greater than `prefix_seq_len` (%d), but is %d instead.\n",
            prefix_seq_len,
            max_seq_len
        );
        GGML_ASSERT(prefix_seq_len < max_seq_len);
    }

    return max_seq_len;
}

void _fan_out_encoder_output(
    ggml_context* ctx,
    ggml_tensor** encoder_output_out,
    ggml_tensor** encoder_padding_mask_out,
    int beam_size
) {
    // (S_enc, M)
    ggml_tensor* encoder_output = *encoder_output_out;
    ggml_tensor* encoder_padding_mask = *encoder_padding_mask_out;

    // (B, S_enc, M)
    ggml_tensor* shape = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, encoder_output->ne[0], encoder_output->ne[1], beam_size);
    // (S_enc, M) -> (B, S_enc, M)
    *encoder_output_out = ggml_repeat(ctx, encoder_output, shape);
    // (S_enc) -> (B, S_enc)
    if (encoder_padding_mask != nullptr) {
        ggml_tensor* shape_mask = ggml_new_tensor_3d(ctx, GGML_TYPE_I8, encoder_padding_mask->ne[0], 1, beam_size);
        *encoder_padding_mask_out = ggml_repeat(ctx, encoder_padding_mask, shape_mask);
    }
}

ggml_tensor* ggml_log_softmax(ggml_context* ctx, ggml_tensor* logits) {
    // TODO: this isn't the most precise way of doing this
    return ggml_log_inplace(ctx, ggml_soft_max_inplace(ctx, logits));
}

ggml_tensor* ggml_expand_2d(ggml_context* ctx, ggml_tensor* x, int64_t ne0, int64_t ne1) {
    ggml_tensor* shape = ggml_new_tensor_2d(ctx, GGML_TYPE_I8, ne0, ne1);
    ggml_type true_type = x->type;
    x->type = GGML_TYPE_F32;
    ggml_tensor* y = ggml_repeat(ctx, x, shape);
    y->type = true_type;
    return y;
}

extern "C" void _bootstrap_seqs_and_scores(
    fairseq2_model& model,
    const SequenceGeneratorJob& job,
    ggml_tensor* full_seqs,
    ggml_tensor* scores,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask
) {
    int prefix_seq_len = job.prefix_seq->ne[0];
    int max_seq_len = scores->ne[0];
    int beam_size = scores->ne[1];
    GGML_ASSERT(prefix_seq_len > 0);
    if (prefix_seq_len == 1)
        return;

    ggml_context* ctx = model.ctx;

    // full_seqs[:, : prefix_seq_len] = job.prefix_seq;
    full_seqs->type = GGML_TYPE_F32;
    job.prefix_seq->type = GGML_TYPE_F32;
    ggml_tensor* seqs = ggml_slice(ctx, full_seqs, 0, 0, prefix_seq_len);
    seqs = ggml_cpy(ctx, ggml_repeat(ctx, job.prefix_seq, seqs), seqs);

    // We have to bootstrap the model with the already fanned-out encoder
    // output to correctly initialize its incremental state.
    // Note: we don't start decoding the last prefix token just yet.
    seqs = ggml_slice(ctx, seqs, 0, 0, prefix_seq_len - 1);
    seqs->type = GGML_TYPE_I32;

    // Bootstrap the model state with prefix sequence.
    seqs = TransformerEmbeddingFrontend_forward(model, "text_decoder_frontend", seqs);
    ggml_tensor* decoder_output = StandardTransformerDecoder_forward(
        model,
        "text_decoder",
        seqs,
        /*padding_mask*/ nullptr,
        encoder_output,
        encoder_padding_mask
    );
    // TODO state_bag.increment_step(prefix_seq_len - 1)

    // logits, lprobs: (N, S_pfx - 1, V)
    ggml_tensor* logits = Linear_forward(model, "final_proj", decoder_output);
    int vocab_size = logits->ne[0];
    ggml_tensor* lprobs = ggml_log_softmax(ctx, ggml_slice(ctx, logits, 1, 0, 1));

    ggml_cgraph gf = ggml_build_forward(lprobs);
    ggml_graph_compute_with_ctx(ctx, &gf, 1);
    ggml_free(ctx);
    full_seqs->type = GGML_TYPE_I32;
    job.prefix_seq->type = GGML_TYPE_I32;

    // Fetch scores of next steps from "lprobs"
    float p_score = 0;
    for (int i = 1; i < prefix_seq_len; ++i) {
        int p = ggml_get_i32_1d(job.prefix_seq, i);
        p_score += ggml_get_f32_1d(lprobs, i * vocab_size + p);
        for (int b = 0; b < beam_size; ++b) {
            // scores: (N, S)
            // Note: First step (e.g. BOS)'s score is always 0.
            ggml_set_f32_1d(scores, b * max_seq_len + i, p_score);
        }
    }
}

/// Finds the topk indices, and write the winning indices in "candidate_indices" array.
int topk(
    ggml_tensor* lprobs,  // (B, V)
    std::int64_t k,
    ggml_tensor* candidate_indices
) {
        // Take the best 2 x `beam_size` predictions. We'll choose the first
    // `beam_size` of these which don't predict EOS to continue with.
    // (N, 2 x B)
    // `vocab_size` - 1 to never select PAD.
    std::int64_t K = std::min(k, ggml_nelements(lprobs));
    auto comp = [lprobs](std::int32_t a, std::int32_t b) {
        return ggml_get_f32_1d(lprobs, a) > ggml_get_f32_1d(lprobs, b);
    };
    GGML_ASSERT(ggml_nelements(candidate_indices) >= k);
    auto cand = (std::int32_t*)candidate_indices->data;
    std::partial_sort(cand, cand + K, cand + ggml_nelements(lprobs), comp);

    return K;
}

void _tweak_lprobs(const SequenceGeneratorJob& job, ggml_tensor* lprobs, int step_nr, int max_seq_len, std::size_t vocab_size) {
        std::size_t beam_size = job.opts.beam_size;
    std::size_t eos_idx = job.eos_idx;

    // Do not allow EOS before reaching the minimum sequence length.
    if (step_nr < job.opts.min_seq_len) {
        // lprobs[:, :, self.eos_idx] = -INFINITY;
        for (size_t i = 0; i < beam_size; ++i)
            ggml_set_f32_1d(lprobs, vocab_size * i + eos_idx, -INFINITY);
    }

    // If we have reached the maximum length, force the last step to be EOS.
    if (step_nr == max_seq_len - 2) {
        // lprobs[:, :, : self.eos_idx]       = -torch.inf
        // lprobs[:, :,   self.eos_idx + 1 :] = -torch.inf
        for (size_t b = 0; b < beam_size; ++b) {
            size_t t = 0;
            for (t = 0; t < eos_idx; ++t)
                ggml_set_f32_1d(lprobs, vocab_size * b + t, -INFINITY);
            for (t = eos_idx + 1; t < vocab_size; ++t)
                ggml_set_f32_1d(lprobs, vocab_size * b + t, -INFINITY);
        }
    }

    // Never allow PAD.
    std::size_t pad_idx = job.pad_idx;
    for (size_t i = 0; i < beam_size; ++i)
        ggml_set_f32_1d(lprobs, vocab_size * i + pad_idx, -INFINITY);

    // Apply UNK penalty.
    if (job.unk_idx >= 0 && job.opts.unk_penalty != 0) {
        // lprobs[:, :, self.unk_idx] -= self.opts.unk_penalty
        auto lprobs_raw = ggml_get_data_f32(lprobs);
        for (size_t i = 0; i < beam_size; ++i)
            lprobs_raw[vocab_size * i + job.unk_idx] -= job.opts.unk_penalty;
    }
}



/// Copies the sequence and scores of a given candidate beam.
void _finalize_hypothesis(
    const SequenceGeneratorJob& job,
    ggml_context* ctx,
    int step_nr,
    std::int32_t beam,
    std::int32_t token,
    float eos_score,
    ggml_tensor* seqs, // (beam_size, seq_len)
    ggml_tensor* scores, // (beam_size, seq_len)
    Hypothesis* hypothesis
) {
        ggml_tensor* seq = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, step_nr + 2);
    hypothesis->seq = seq;
    ggml_tensor* step_scores = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, step_nr + 2);
    hypothesis->step_scores = step_scores;

    auto tok = (std::int32_t*)seq->data;
    for (int i = 0; i < step_nr + 1; ++i) {
        tok[i] = ggml_get_i32_1d(seqs, seqs->ne[0] * beam + i);
    }
    tok[step_nr + 1] = token;

    // Convert from cumulative to per-step scores.
    auto sc = (float*)step_scores->data;
    float last_score = eos_score;
    for (int i = step_nr; i >= 0; --i) {
        float sc0 = ggml_get_f32_1d(scores, scores->ne[0] * beam + i);
        sc[i + 1] = last_score - sc0;
        last_score = sc0;
    }
    sc[0] = 0;

    if (job.opts.normalize_scores)
        // Skip first EOS since it is always 0 and skews normalization.
        eos_score /= (float)std::pow((step_nr + 1), job.opts.len_penalty);
    hypothesis->score = eos_score;
}

// Uses ggml_context to store any object.
#define GGML_CTX_ALLOC(ctx, Type, n) \
    (Type*)(ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Type) * n)->data);


ggml_context* ctx_from_buffer(std::vector<uint8_t>& buffer) {
    return ggml_init({
        /*.mem_size   =*/ static_cast<int64_t>(buffer.capacity()),
        /*.mem_buffer =*/ buffer.data(),
        /*.no_alloc   =*/ false,
    });
}


/// Generates a translation for a single sequence
// TODO: clean ups
// * replace manual tensor tweaking with ggml_set_*d (a ggml_set_slice could be useful)
extern "C" Hypothesis* generate_sequence(
    fairseq2_model& model,
    const SequenceGeneratorJob& job,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask,
    ggml_context* result_ctx
) {
    std::vector<uint8_t> local_bufs[3] = {
        std::vector<uint8_t>(1024 * 1024 * 1024),  // step_ctx
        std::vector<uint8_t>(1024 * 1024 * 1024),  // next_step_ctx
        std::vector<uint8_t>(1024 * 1024 * 1024)  // search_ctx
    };
    ggml_context* search_ctx = ctx_from_buffer(local_bufs[2]);

    ggml_tensor* embed = model.tensors["text_decoder_frontend.embed.weight"];
    size_t vocab_size = embed->ne[1];
    std::size_t beam_size = job.opts.beam_size;
    int source_seq_len = encoder_output->ne[1];
    int max_seq_len = _determine_max_seq_len(job, source_seq_len);

    ggml_context* original_ctx = model.ctx;
    model.ctx = search_ctx;
    fairseq2_kv_cache_alloc(model, beam_size, max_seq_len);

    // (S_enc, M) -> (B, S_enc, M)
    _fan_out_encoder_output(search_ctx, &encoder_output, &encoder_padding_mask, beam_size);

    // Allocate results in the context provided by the caller.
    Hypothesis* finished_searches_begin = GGML_CTX_ALLOC(result_ctx, Hypothesis, beam_size);
    Hypothesis* finished_searches = finished_searches_begin;
    for (std::size_t i = 0; i < beam_size; ++i) finished_searches[i] = {nullptr, -INFINITY, nullptr};
    Hypothesis* finished_searches_end = finished_searches + beam_size;

    // Initialize buffers. (B, S)
    ggml_tensor* seqs = ggml_new_tensor_2d(search_ctx, GGML_TYPE_I32, max_seq_len, beam_size);
    ggml_set_i32(seqs, 0);
    ggml_set_name(seqs, "seqs_0");
    ggml_tensor* scores = ggml_new_tensor_2d(search_ctx, GGML_TYPE_F32, max_seq_len, beam_size);
    ggml_set_name(scores, "scores_0");
    ggml_set_f32(scores, 0.0);

    _bootstrap_seqs_and_scores(
        model, job, seqs, scores, encoder_output, encoder_padding_mask
    );
    int prefix_seq_len = job.prefix_seq->ne[0];
    int start_step = prefix_seq_len - 1;

    // Holds the indices of beams (a beam can occur more than once) that we
    // should continue with in the next step.
    ggml_tensor* beam_indices = ggml_new_tensor_1d(search_ctx, GGML_TYPE_I32, beam_size);
    ggml_tensor* next_tokens = ggml_new_tensor_1d(search_ctx, GGML_TYPE_I32, beam_size);
    ggml_tensor* next_scores = ggml_new_tensor_1d(search_ctx, GGML_TYPE_F32, beam_size);

    // Array with integers up to 'vocab_size * beam_size' to represent next beams to explore
    ggml_tensor* candidate_indices = ggml_new_tensor_1d(search_ctx, GGML_TYPE_I32, vocab_size * beam_size);
    for (std::size_t i = 0; i < vocab_size * beam_size; ++i)
        ((int32_t *)(candidate_indices->data))[i] = i;

    printf_mem_usage(search_ctx, "search_ctx");

    ggml_context* step_ctx = ctx_from_buffer(local_bufs[0]);
    ggml_context* next_step_ctx = nullptr;
    for (int step_nr = start_step; step_nr < max_seq_len - 1; ++step_nr) {
        model.ctx = step_ctx;
        ggml_tensor* prev_token = ggml_slice(step_ctx, seqs, 0, step_nr, step_nr + 1);
        ggml_tensor* decoder_input = TransformerEmbeddingFrontend_forward(model, "text_decoder_frontend", prev_token);
        ggml_tensor* decoder_output = StandardTransformerDecoder_forward(
            model,
            "text_decoder",
            decoder_input,
            nullptr,  // We never generate PAD.
            encoder_output,
            encoder_padding_mask
        ); // (B, 1, D)

        // Just look at the last token.
        decoder_output = ggml_flatten_1d(step_ctx, decoder_output, 0);  // (B, model_dim)
        ggml_tensor* logits = Linear_forward(model, "final_proj", decoder_output);  // (B, vocab_size)
        ggml_tensor* lprobs = ggml_log_softmax(step_ctx, logits);

        // Compute lprobs here so we can modify it in place in the lprob tweaking phase
        // TODO: use ggml properly compute the tweaks
        ggml_cgraph gf = ggml_build_forward(lprobs);
        // printf("beam search step %d. Graph.n_nodes: %d\n", step_nr, gf.n_nodes);
        ggml_graph_compute_with_ctx(step_ctx, &gf, 1);
        ggml_detach(lprobs);

        _tweak_lprobs(job, lprobs, step_nr, max_seq_len, vocab_size);

        ggml_tensor* last_scores = ggml_slice(step_ctx, scores, 0, step_nr, step_nr+1);
        if (step_nr == start_step) {
            // At the initial step, all hypotheses are equally likely, so we use
            // only the first beam.
            lprobs = ggml_slice(step_ctx, lprobs, 1, 0, 1);
            lprobs = ggml_cont(step_ctx, lprobs);
            // The first step always indicates the beginning of the sequence and has no score.
            if (step_nr > 0) {
                last_scores = ggml_slice(step_ctx, last_scores, 1, 0, 1);
                lprobs = ggml_add_inplace(step_ctx, lprobs, ggml_repeat(step_ctx, last_scores, lprobs));
            }
        } else {
            // Make probabilities contain cumulative scores for each hypothesis.
            lprobs = ggml_add_inplace(step_ctx, lprobs, ggml_repeat(step_ctx, last_scores, lprobs));
        }

        gf = ggml_build_forward(lprobs);
        ggml_graph_compute_with_ctx(step_ctx, &gf, 1);

        // Determine (beam, token) candidates for the next step.
        // (N, 2 x B)
        std::int64_t K = topk(
            lprobs, std::min(2 * beam_size, vocab_size - 1), candidate_indices
        );

        std::size_t ongoing_beams = 0;
        for (std::int32_t i = 0; i < K; ++i) {
            int c = ggml_get_f32_1d(candidate_indices, i);
            std::int32_t beam = c / vocab_size;
            std::int32_t token = c % vocab_size;
            float tok_score = ggml_get_f32_1d(lprobs, c);

            // Detect beams that reached the minimum length and that end with an EOS.
            bool eos = token == job.eos_idx;
            eos &= tok_score != -INFINITY;
            if (eos) {
                _finalize_hypothesis(job, result_ctx, step_nr, beam, token, tok_score, seqs, scores, finished_searches++);
                if (finished_searches == finished_searches_end)
                    goto end_of_beam_search;
                continue;
            }

            ggml_set_f32_1d(beam_indices, ongoing_beams, beam);
            ggml_set_f32_1d(next_tokens, ongoing_beams, token);
            ggml_set_f32_1d(next_scores, ongoing_beams, tok_score);
            ongoing_beams += 1;
            if (ongoing_beams >= beam_size) break;
        }

        // Reorder beams in the `seq` and `score` buffers. The same beam can
        // be selected more than once.
        ggml_tensor* new_seqs = seqs;
        ggml_tensor* new_scores = scores;
        if (step_nr > start_step) {
            // (B, S), (B) -> (B, S)
            // ggml_get_rows and ggml_set only work with floats ...
            new_seqs->type = GGML_TYPE_F32;
            new_seqs = ggml_get_rows(search_ctx, seqs, beam_indices);
            new_scores = ggml_get_rows(search_ctx, scores, beam_indices);
            ggml_cgraph gf_reorder = ggml_build_forward(new_seqs);
            ggml_build_forward_expand(&gf_reorder, new_scores);
            reorder_kv_cache(model, step_ctx, &gf_reorder, beam_indices);
            ggml_graph_compute_with_ctx(step_ctx, &gf_reorder, 1);
            ggml_detach(new_seqs);
            ggml_detach(new_scores);
            new_seqs->type = GGML_TYPE_I32;
            printf_mem_usage(search_ctx, "search_ctx");
            next_step_ctx = ctx_from_buffer(local_bufs[(step_nr + 1) % 2]);
            SWAP(step_ctx, next_step_ctx);
            ggml_free(next_step_ctx);
        }

        // new_seqs[:, step_nr + 1] = next_tokens
        // new_scores[:, step_nr + 1] = next_scores
        for (std::size_t i = 0; i < beam_size; ++i) {
            ((std::int32_t*)new_seqs->data)[step_nr + 1 + i * max_seq_len] = ggml_get_i32_1d(next_tokens, i);
            ((float*)new_scores->data)[step_nr + 1 + i * max_seq_len] = ggml_get_f32_1d(next_scores, i);
        }

        // TODO the old seqs and score buffers could be reused for next step
        seqs = new_seqs;
        scores = new_scores;
        printf_mem_usage(step_ctx, "step_ctx");
    }

end_of_beam_search:
    // Ensure that hypotheses are sorted by decreasing scores before returning.
    std::sort(
        finished_searches_begin,
        finished_searches_end,
        [](Hypothesis a, Hypothesis b) { return a.score > b.score; }
    );

    fairseq2_kv_cache_reset(model);
    model.ctx = original_ctx;
    return finished_searches_begin;
}

extern "C" Hypothesis* _testing_return_hypothesis_ptr(ggml_context* ctx) {
    Hypothesis* result = GGML_CTX_ALLOC(ctx, struct Hypothesis, 2);

    result[0] = {ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), 3.14f, (ggml_tensor*)result};
    ggml_set_i32_1d(result[0].seq, 0, 314);

    result[1] = {ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1), 4.21f, nullptr};
    ggml_set_i32_1d(result[1].seq, 0, 421);

    return result;
}

// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4



struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
    llama_vocab::id id;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
    llama_vocab::id id;
};

struct llm_tokenizer_spm {
    llm_tokenizer_spm(const llama_vocab & vocab): vocab(vocab) {}

    void tokenize(const std::string& input_text, ggml_tensor& output) {
        llama_vocab::id unk_idx = vocab.token_to_id.at("<unk>");

        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        // This is kind of annoying, but needed because with SPM,
        // characters following a space have a special meaning.
        // And the algorithm rely on substrings to do the lookups.
        std::string text = input_text;
        bool need_extra_space = text.size() > 0 && text[0] != ' ';
        if (need_extra_space) text = " " + text;

        while (offs < text.size()) {
            size_t len = utf8_len(text[offs]);
            size_t n = std::min(len, text.size() - offs);

            auto token = vocab.token_to_id.find(std::string(text, offs, n));
            llama_vocab::id id = token == vocab.token_to_id.end() ? unk_idx : token->second;
            llm_symbol sym = {
                /*prev*/ index - 1,
                /*next*/ offs + n == text.size() ? -1 : index + 1,
                /*text*/ text.c_str() + offs,
                /*n*/ n,
                /*id*/ id
            };
            offs += n;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];
            const std::string text = std::string(left_sym.text, left_sym.n + right_sym.n);

            // if one of the symbols already got merged, skip it.
            if (
                left_sym.n == 0
                || right_sym.n == 0
                || left_sym.n + right_sym.n != bigram.size
            ) continue;

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            left_sym.id = bigram.id;
            right_sym.n = 0;

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        llama_vocab::id* out = (llama_vocab::id*)output.data;
        int out_step = sizeof(llama_vocab::id) / output.nb[0];
        int num_tokens = 0;
        for (int i = 0; i > -1; i = symbols[i].next) {
            llm_symbol& symbol = symbols[i];
            *(out + num_tokens * out_step) = symbol.id;
            num_tokens += 1;
        }
        *(out + num_tokens * out_step) = vocab.token_to_id.at("</s>");
        num_tokens += 1;
        output.ne[0] = num_tokens;
    }

private:

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.token_to_id.find(text);

        if (token == vocab.token_to_id.end()) {
            return;
        }

        llama_vocab::id id = token->second;
        if (static_cast<size_t>(id) >= vocab.id_to_token.size()) {
            return;
        }

        const auto& tok_data = vocab.id_to_token[id];
        llm_bigram_spm bigram = {
            /*left */ left,
            /*right*/ right,
            /*score*/ tok_data.score,
            /*size */ text.size(),
            /*id */ id
        };
        work_queue.push(bigram);
    }

    const llama_vocab& vocab;
    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
};


extern "C" void fairseq2_spm_tokenize(fairseq2_model* model, const char* text, ggml_tensor& out) {
    llm_tokenizer_spm spm = {model->vocab};
    spm.tokenize(std::string(text), out);
}

extern "C" std::size_t fairseq2_spm_detokenize(fairseq2_model* model, ggml_tensor* tokens, char* out) {
    int eos_idx = model->vocab.token_to_id["</s>"];
    int sent_len = tokens->ne[0];
    std::size_t written = 0;
    for (int i = 0; i < sent_len; ++i) {
        int id = ggml_get_i32_1d(tokens, i);
        // Don't print the EOS token but only if it appear at the end.
        if (i == sent_len - 1 && eos_idx == id) break;

        std::string token = model->vocab.id_to_token.at(id).text;
        // Skip the first space outputted.
        auto begin = token.begin();
        if (i == 0 && token.size() > 0 && token[0] == ' ') begin += 1;
        std::copy(begin, token.end(), out);
        std::size_t n = token.end() - begin;
        written += n;
        out += n;
    }
    *out = '0';
    return written;
}
