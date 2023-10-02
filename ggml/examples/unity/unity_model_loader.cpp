// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "common.h"
#include "common-ggml.h"

#include "unity_model_loader.h"

void unity_model_loader::load_hparams(fairseq2_model& model, std::ifstream &fin)
{
    unity_hparams* hparams = (unity_hparams*)model.hparams;
    read_unity_hparams(hparams, fin);
    if (hparams->__end_of_hparams__ != 6877961321223123048) {
        throw std::invalid_argument("");
    }
}

std::size_t
unity_model_loader::compute_context_size(void* raw_hparams)
{
    auto* hparams = (unity_hparams*)raw_hparams;
    return hparams->model_byte_size;
};

struct UnityArch {
    struct TransformerDecoder text_decoder;
};

void unity_model_loader::tensors_alloc(fairseq2_model &model)
{
    auto hparams = (unity_hparams&)model.hparams;
    auto& arch = (UnityArch&)model.arch;
    const auto ctx = model.ctx;
    auto tensors = model.tensors;

    const auto vocab_size = hparams.nllb_config__vocabulary_size;
    const auto model_dim = hparams.nllb_config__model_dim;

    // This can be simplified by adding syntax sugar

    // frontend
    // arch.frontend_embed_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, dim);
    // tensor_map["text_decoder_frontend.embed.weight"] = arch.frontend_embed_w;

    // layers
    {
        const auto n_layers = hparams.nllb_config__num_decoder_layers;
        arch.text_decoder.layers = std::vector<TransformerDecoderLayer>(n_layers);
        auto layers = arch.text_decoder.layers;
        auto num_heads = hparams.nllb_config__num_decoder_attn_heads;
        for (int i = 0; i < n_layers; ++i) {
            auto prefix = "text_decoder.layers." + std::to_string(i);
            MultiheadAttention_init(layers[i].self_attn, model, prefix + "self_attn", model_dim, num_heads);
            LayerNorm_init(layers[i].self_attn_norm, model, prefix + "self_attn_norm", model_dim);
        }
    }

    // // layer_norm
    // arch.layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    // tensor_map["text_decoder.layer_norm.weight"] = arch.layer_norm_w;
    // arch.layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
    // tensor_map["text_decoder.layer_norm.bias"] = arch.layer_norm_b;
};

extern "C" void load_unity_ggml_file(fairseq2_model& model, const char* fname) {
    return load_fairseq2_ggml_file<unity_model_loader>(model, fname);
}



