// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>
#include "model_loader.h"


// TODO Merge with Ning implementation
struct unity_hparams {
    int32_t model_dim;
    int32_t w2v2_encoder_config__model_dim;
    int32_t w2v2_encoder_config__max_seq_len;
    int32_t w2v2_encoder_config__feature_dim;
    int32_t w2v2_encoder_config__use_fbank;
    float w2v2_encoder_config__first_pass_dropout_p;
    int32_t w2v2_encoder_config__layer_norm_features;
    int32_t w2v2_encoder_config__feature_extractor_bias;
    int32_t w2v2_encoder_config__feature_extractor_layer_norm_convs;
    int32_t w2v2_encoder_config__feature_grad_scale;
    int32_t w2v2_encoder_config__num_fbank_channels;
    int32_t w2v2_encoder_config__fbank_stride;
    int32_t w2v2_encoder_config__sample_fbank_every_k;
    int32_t w2v2_encoder_config__pos_encoder_depth;
    int32_t w2v2_encoder_config__pos_conv_kernel_size;
    int32_t w2v2_encoder_config__num_pos_conv_groups;
    int32_t w2v2_encoder_config__use_conformer;
    int32_t w2v2_encoder_config__num_encoder_layers;
    int32_t w2v2_encoder_config__num_encoder_attn_heads;
    int32_t w2v2_encoder_config__ffn_inner_dim;
    float w2v2_encoder_config__dropout_p;
    float w2v2_encoder_config__attn_dropout_p;
    float w2v2_encoder_config__layer_drop_p;
    int32_t w2v2_encoder_config__norm_order;
    int32_t w2v2_encoder_config__depthwise_conv_kernel_size;
    int32_t nllb_config__model_dim;
    int32_t nllb_config__max_seq_len;
    int32_t nllb_config__vocabulary_size;
    int32_t nllb_config__pad_idx;
    int32_t nllb_config__num_encoder_layers;
    int32_t nllb_config__num_decoder_layers;
    int32_t nllb_config__num_encoder_attn_heads;
    int32_t nllb_config__num_decoder_attn_heads;
    int32_t nllb_config__ffn_inner_dim;
    float nllb_config__dropout_p;
    int32_t t2u_config__model_dim;
    int32_t t2u_config__unit_max_seq_len;
    int32_t t2u_config__unit_vocabulary_size;
    int32_t t2u_config__unit_pad_idx;
    int32_t t2u_config__num_encoder_layers;
    int32_t t2u_config__num_decoder_layers;
    int32_t t2u_config__num_encoder_attn_heads;
    int32_t t2u_config__num_decoder_attn_heads;
    int32_t t2u_config__ffn_inner_dim;
    float t2u_config__dropout_p;
    int32_t use_text_encoder;
    int32_t use_conformer_adaptor;
    int32_t num_adaptor_layers;
    int32_t adaptor_kernel_size;
    int32_t adaptor_stride;
    int32_t adaptor_layer_norm;
    float adaptor_dropout_p;
};

// Methods

// Embedding
std::size_t compute_embed_size(int32_t vocab_size, int32_t dim)
{
    return vocab_size * dim * ggml_type_size(GGML_TYPE_F32);
};

// Projection
std::size_t compute_projection_size(int32_t in_dim, int32_t out_dim)
{
    return (in_dim * out_dim * ggml_type_size(GGML_TYPE_F32)) // weight
        + (out_dim * ggml_type_size(GGML_TYPE_F32)); // bias
};

// LayerNorm
std::size_t compute_layer_norm_size(int32_t dim)
{
    return 2 * dim * ggml_type_size(GGML_TYPE_F32); // weight and bias
};

// FFN Layer

struct ffn_layer {
    struct ggml_tensor* layer_norm_w; // model_dim
    struct ggml_tensor* layer_norm_b; // model_dim

    struct ggml_tensor* inner_proj_w; // ffn_inner_dim x model_dim
    struct ggml_tensor* inner_proj_b; // ffn_inner_dim

    struct ggml_tensor* output_proj_w; // model_dim x ffn_inner_dim
    struct ggml_tensor* output_proj_b; // model_dim
};

std::size_t compute_ffn_layer_size(int32_t dim, int32_t inner_dim)
{
    return compute_layer_norm_size(dim)
        + compute_projection_size(dim, inner_dim)
        + compute_projection_size(inner_dim, dim);
};

void init_ffn_layer(
    ffn_layer *layer,
    fairseq2_model<unity_hparams> &model_ctx,
    const std::string &prefix)
{
    const auto dim = model_ctx.hparams.nllb_config__model_dim;
    const auto inner_dim = model_ctx.hparams.nllb_config__ffn_inner_dim;
    auto ctx = model_ctx.ctx;
    auto &tensor_map = model_ctx.tensors;

    layer->layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + "_layer_norm.weight"] = layer->layer_norm_w;
    layer->layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + "_layer_norm.bias"] = layer->layer_norm_b;

    layer->inner_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, inner_dim, dim);
    tensor_map[prefix + ".inner_proj.weight"] = layer->inner_proj_w;
    layer->inner_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, inner_dim);
    tensor_map[prefix + ".inner_proj.bias"] = layer->inner_proj_b;

    layer->output_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, inner_dim);
    tensor_map[prefix + ".output_proj.weight"] = layer->output_proj_w;
    layer->output_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + ".output_proj.bias"] = layer->output_proj_b;
}

// Attention Layer

struct attention_layer {
    struct ggml_tensor* layer_norm_w; // model_dim
    struct ggml_tensor* layer_norm_b; // model_dim

    struct ggml_tensor* q_proj_w; // model_dim x model_dim
    struct ggml_tensor* q_proj_b; // model_dim
    struct ggml_tensor* k_proj_w; // model_dim x model_dim
    struct ggml_tensor* k_proj_b; // model_dim
    struct ggml_tensor* v_proj_w; // model_dim x model_dim
    struct ggml_tensor* v_proj_b; // model_dim

    struct ggml_tensor* output_proj_w; // model_dim x model_dim
    struct ggml_tensor* output_proj_b; // model_dim
};

std::size_t compute_attention_layer_size(int32_t dim)
{
    return compute_layer_norm_size(dim)
        + 4 * compute_projection_size(dim, dim); // q, k, v, and out
};

void init_attention_layer(
    attention_layer *layer,
    fairseq2_model<unity_hparams> &model_ctx,
    const std::string &prefix)
{
    const auto dim = model_ctx.hparams.nllb_config__model_dim;
    auto ctx = model_ctx.ctx;
    auto &tensor_map = model_ctx.tensors;

    layer->layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + "_layer_norm.weight"] = layer->layer_norm_w;
    layer->layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + "_layer_norm.bias"] = layer->layer_norm_b;

    layer->q_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
    tensor_map[prefix + ".q_proj.weight"] = layer->q_proj_w;
    layer->q_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + ".q_proj.bias"] = layer->q_proj_b;

    layer->k_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
    tensor_map[prefix + ".k_proj.weight"] = layer->k_proj_w;
    layer->k_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + ".k_proj.bias"] = layer->k_proj_b;

    layer->v_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
    tensor_map[prefix + ".v_proj.weight"] = layer->v_proj_w;
    layer->v_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + ".v_proj.bias"] = layer->v_proj_b;

    layer->output_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, dim);
    tensor_map[prefix + ".output_proj.weight"] = layer->output_proj_w;
    layer->output_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map[prefix + ".output_proj.bias"] = layer->output_proj_b;
}


// Attention Head

struct attention_head {
    struct attention_layer* self_attn; // model_dim
    struct attention_layer* encoder_decoder_attn; // model_dim
    struct ffn_layer* ffn;
};

std::size_t compute_attention_head_size(int32_t dim, int32_t inner_dim)
{
    return 2 * compute_attention_layer_size(dim)
        + compute_ffn_layer_size(dim, inner_dim);
};

void init_attention_head(
    attention_head *head,
    fairseq2_model<unity_hparams> &model_ctx,
    const std::string &prefix)
{
    init_attention_layer(head->self_attn, model_ctx, prefix + ".self_attn");
    init_attention_layer(head->encoder_decoder_attn, model_ctx, prefix + ".encoder_decoder_attn");
    init_ffn_layer(head->ffn, model_ctx, prefix + ".ffn");
}

// TODO: attention_head_compute_graph

// Text Decoder

struct text_decoder {
    struct ggml_tensor* frontend_embed_w; // vocab_size x model_dim
    std::vector<attention_head*> multi_head;
    struct ggml_tensor* layer_norm_w;
    struct ggml_tensor* layer_norm_b;
};

std::size_t compute_context_size(unity_hparams &hparams)
{
    const auto vocab_size = hparams.nllb_config__vocabulary_size;
    const auto dim = hparams.nllb_config__model_dim;
    const auto inner_dim = hparams.nllb_config__ffn_inner_dim;
    const auto n_layers = hparams.nllb_config__num_decoder_layers;

    const auto overhead = (6 + 12 * n_layers) * 512; // TODO Find out what this is.

    return compute_embed_size(vocab_size, dim)
        + n_layers * compute_attention_head_size(dim, inner_dim)
        + compute_layer_norm_size(dim)
        + overhead;
};

void init_model_tensors(
    text_decoder &model,
    fairseq2_model<unity_hparams> &model_ctx,
    const std::string &prefix)
{
    const auto ctx = model_ctx.ctx;
    const auto hparams = model_ctx.hparams;
    auto tensor_map = model_ctx.tensors;

    const auto vocab_size = hparams.nllb_config__vocabulary_size;
    const auto dim = hparams.nllb_config__model_dim;
    const auto n_layers = hparams.nllb_config__num_decoder_layers;

    // This can be simplified by adding syntax sugar

    // frontend
    model.frontend_embed_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, dim);
    tensor_map["text_decoder_frontend.embed.weight"] = model.frontend_embed_w;

    // layers
    model.multi_head.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        auto head = model.multi_head[i];
        auto prefix = "text_decoder.layers." + std::to_string(i);
        init_attention_head(head, model_ctx, prefix);
    }

    // layer_norm
    model.layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map["text_decoder.layer_norm.weight"] = model.layer_norm_w;
    model.layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    tensor_map["text_decoder.layer_norm.bias"] = model.layer_norm_b;
};



// Model
class unity_model_loader: public model_loader<unity_hparams> {
protected:
    void
    load_hparams(std::ifstream &fin, unity_hparams &hparams);

    std::size_t
    compute_context_size(unity_hparams &hparams) = 0;

    void
    init_model_tensors(fairseq2_model<unity_hparams> &model);
};
