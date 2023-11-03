// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <vector>
#include "model_loader.h"


struct unity_hparams {
    std::int64_t model_dim;
    std::int64_t w2v2_encoder_config__model_dim;
    std::int64_t w2v2_encoder_config__max_seq_len;
    std::int64_t w2v2_encoder_config__feature_dim;
    std::int64_t w2v2_encoder_config__use_fbank;
    double w2v2_encoder_config__first_pass_dropout_p;
    std::int64_t w2v2_encoder_config__layer_norm_features;
    // Error: Unsupported type <class 'list'> w2v2_encoder_config__feature_extractor_layer_descs;
    std::int64_t w2v2_encoder_config__feature_extractor_bias;
    std::int64_t w2v2_encoder_config__feature_extractor_layer_norm_convs;
    std::int64_t w2v2_encoder_config__feature_grad_scale;
    std::int64_t w2v2_encoder_config__num_fbank_channels;
    std::int64_t w2v2_encoder_config__fbank_stride;
    std::int64_t w2v2_encoder_config__sample_fbank_every_k;
    // Error: Unsupported type <class 'str'> w2v2_encoder_config__pos_encoder_type;
    std::int64_t w2v2_encoder_config__pos_encoder_depth;
    std::int64_t w2v2_encoder_config__pos_conv_kernel_size;
    std::int64_t w2v2_encoder_config__num_pos_conv_groups;
    std::int64_t w2v2_encoder_config__use_conformer;
    std::int64_t w2v2_encoder_config__num_encoder_layers;
    std::int64_t w2v2_encoder_config__num_encoder_attn_heads;
    std::int64_t w2v2_encoder_config__ffn_inner_dim;
    double w2v2_encoder_config__dropout_p;
    double w2v2_encoder_config__attn_dropout_p;
    double w2v2_encoder_config__layer_drop_p;
    std::int64_t w2v2_encoder_config__norm_order;
    std::int64_t w2v2_encoder_config__depthwise_conv_kernel_size;
    std::int64_t mt_model_config__model_dim;
    std::int64_t mt_model_config__max_seq_len;
    std::int64_t mt_model_config__vocabulary_size;
    std::int64_t mt_model_config__pad_idx;
    std::int64_t mt_model_config__num_encoder_layers;
    std::int64_t mt_model_config__num_decoder_layers;
    std::int64_t mt_model_config__num_encoder_attn_heads;
    std::int64_t mt_model_config__num_decoder_attn_heads;
    std::int64_t mt_model_config__ffn_inner_dim;
    double mt_model_config__dropout_p;
    std::int64_t t2u_config__model_dim;
    std::int64_t t2u_config__unit_max_seq_len;
    std::int64_t t2u_config__unit_vocabulary_size;
    std::int64_t t2u_config__unit_pad_idx;
    std::int64_t t2u_config__num_encoder_layers;
    std::int64_t t2u_config__num_decoder_layers;
    std::int64_t t2u_config__num_encoder_attn_heads;
    std::int64_t t2u_config__num_decoder_attn_heads;
    std::int64_t t2u_config__ffn_inner_dim;
    double t2u_config__dropout_p;
    std::int64_t use_text_encoder;
    std::int64_t use_conformer_adaptor;
    std::int64_t num_adaptor_layers;
    std::int64_t adaptor_kernel_size;
    std::int64_t adaptor_stride;
    std::int64_t adaptor_layer_norm;
    double adaptor_dropout_p;
    std::int64_t model_byte_size;
    std::int64_t __end_of_hparams__;
};

void read_unity_hparams(unity_hparams& out, std::ifstream &fin) {
    fin.read((char*) &out.model_dim, sizeof(out.model_dim));
    fin.read((char*) &out.w2v2_encoder_config__model_dim, sizeof(out.w2v2_encoder_config__model_dim));
    fin.read((char*) &out.w2v2_encoder_config__max_seq_len, sizeof(out.w2v2_encoder_config__max_seq_len));
    fin.read((char*) &out.w2v2_encoder_config__feature_dim, sizeof(out.w2v2_encoder_config__feature_dim));
    fin.read((char*) &out.w2v2_encoder_config__use_fbank, sizeof(out.w2v2_encoder_config__use_fbank));
    fin.read((char*) &out.w2v2_encoder_config__first_pass_dropout_p, sizeof(out.w2v2_encoder_config__first_pass_dropout_p));
    fin.read((char*) &out.w2v2_encoder_config__layer_norm_features, sizeof(out.w2v2_encoder_config__layer_norm_features));
    fin.read((char*) &out.w2v2_encoder_config__feature_extractor_bias, sizeof(out.w2v2_encoder_config__feature_extractor_bias));
    fin.read((char*) &out.w2v2_encoder_config__feature_extractor_layer_norm_convs, sizeof(out.w2v2_encoder_config__feature_extractor_layer_norm_convs));
    fin.read((char*) &out.w2v2_encoder_config__feature_grad_scale, sizeof(out.w2v2_encoder_config__feature_grad_scale));
    fin.read((char*) &out.w2v2_encoder_config__num_fbank_channels, sizeof(out.w2v2_encoder_config__num_fbank_channels));
    fin.read((char*) &out.w2v2_encoder_config__fbank_stride, sizeof(out.w2v2_encoder_config__fbank_stride));
    fin.read((char*) &out.w2v2_encoder_config__sample_fbank_every_k, sizeof(out.w2v2_encoder_config__sample_fbank_every_k));
    fin.read((char*) &out.w2v2_encoder_config__pos_encoder_depth, sizeof(out.w2v2_encoder_config__pos_encoder_depth));
    fin.read((char*) &out.w2v2_encoder_config__pos_conv_kernel_size, sizeof(out.w2v2_encoder_config__pos_conv_kernel_size));
    fin.read((char*) &out.w2v2_encoder_config__num_pos_conv_groups, sizeof(out.w2v2_encoder_config__num_pos_conv_groups));
    fin.read((char*) &out.w2v2_encoder_config__use_conformer, sizeof(out.w2v2_encoder_config__use_conformer));
    fin.read((char*) &out.w2v2_encoder_config__num_encoder_layers, sizeof(out.w2v2_encoder_config__num_encoder_layers));
    fin.read((char*) &out.w2v2_encoder_config__num_encoder_attn_heads, sizeof(out.w2v2_encoder_config__num_encoder_attn_heads));
    fin.read((char*) &out.w2v2_encoder_config__ffn_inner_dim, sizeof(out.w2v2_encoder_config__ffn_inner_dim));
    fin.read((char*) &out.w2v2_encoder_config__dropout_p, sizeof(out.w2v2_encoder_config__dropout_p));
    fin.read((char*) &out.w2v2_encoder_config__attn_dropout_p, sizeof(out.w2v2_encoder_config__attn_dropout_p));
    fin.read((char*) &out.w2v2_encoder_config__layer_drop_p, sizeof(out.w2v2_encoder_config__layer_drop_p));
    fin.read((char*) &out.w2v2_encoder_config__norm_order, sizeof(out.w2v2_encoder_config__norm_order));
    fin.read((char*) &out.w2v2_encoder_config__depthwise_conv_kernel_size, sizeof(out.w2v2_encoder_config__depthwise_conv_kernel_size));
    fin.read((char*) &out.mt_model_config__model_dim, sizeof(out.mt_model_config__model_dim));
    fin.read((char*) &out.mt_model_config__max_seq_len, sizeof(out.mt_model_config__max_seq_len));
    fin.read((char*) &out.mt_model_config__vocabulary_size, sizeof(out.mt_model_config__vocabulary_size));
    fin.read((char*) &out.mt_model_config__pad_idx, sizeof(out.mt_model_config__pad_idx));
    fin.read((char*) &out.mt_model_config__num_encoder_layers, sizeof(out.mt_model_config__num_encoder_layers));
    fin.read((char*) &out.mt_model_config__num_decoder_layers, sizeof(out.mt_model_config__num_decoder_layers));
    fin.read((char*) &out.mt_model_config__num_encoder_attn_heads, sizeof(out.mt_model_config__num_encoder_attn_heads));
    fin.read((char*) &out.mt_model_config__num_decoder_attn_heads, sizeof(out.mt_model_config__num_decoder_attn_heads));
    fin.read((char*) &out.mt_model_config__ffn_inner_dim, sizeof(out.mt_model_config__ffn_inner_dim));
    fin.read((char*) &out.mt_model_config__dropout_p, sizeof(out.mt_model_config__dropout_p));
    fin.read((char*) &out.t2u_config__model_dim, sizeof(out.t2u_config__model_dim));
    fin.read((char*) &out.t2u_config__unit_max_seq_len, sizeof(out.t2u_config__unit_max_seq_len));
    fin.read((char*) &out.t2u_config__unit_vocabulary_size, sizeof(out.t2u_config__unit_vocabulary_size));
    fin.read((char*) &out.t2u_config__unit_pad_idx, sizeof(out.t2u_config__unit_pad_idx));
    fin.read((char*) &out.t2u_config__num_encoder_layers, sizeof(out.t2u_config__num_encoder_layers));
    fin.read((char*) &out.t2u_config__num_decoder_layers, sizeof(out.t2u_config__num_decoder_layers));
    fin.read((char*) &out.t2u_config__num_encoder_attn_heads, sizeof(out.t2u_config__num_encoder_attn_heads));
    fin.read((char*) &out.t2u_config__num_decoder_attn_heads, sizeof(out.t2u_config__num_decoder_attn_heads));
    fin.read((char*) &out.t2u_config__ffn_inner_dim, sizeof(out.t2u_config__ffn_inner_dim));
    fin.read((char*) &out.t2u_config__dropout_p, sizeof(out.t2u_config__dropout_p));
    fin.read((char*) &out.use_text_encoder, sizeof(out.use_text_encoder));
    fin.read((char*) &out.use_conformer_adaptor, sizeof(out.use_conformer_adaptor));
    fin.read((char*) &out.num_adaptor_layers, sizeof(out.num_adaptor_layers));
    fin.read((char*) &out.adaptor_kernel_size, sizeof(out.adaptor_kernel_size));
    fin.read((char*) &out.adaptor_stride, sizeof(out.adaptor_stride));
    fin.read((char*) &out.adaptor_layer_norm, sizeof(out.adaptor_layer_norm));
    fin.read((char*) &out.adaptor_dropout_p, sizeof(out.adaptor_dropout_p));
    fin.read((char*) &out.model_byte_size, sizeof(out.model_byte_size));
    fin.read((char*) &out.__end_of_hparams__, sizeof(out.__end_of_hparams__));
};

class unity_model_loader: public model_loader {
    public:
    void load_hparams(fairseq2_model& model, std::ifstream &fin);

    std::size_t compute_context_size(void* raw_hparams);
};
