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


void
unity_model_loader::load_hparams(std::ifstream& fin, unity_hparams& hparams)
{
    fin.read((char*) &hparams.model_dim, sizeof(hparams.model_dim));
    fin.read((char*) &hparams.w2v2_encoder_config__model_dim, sizeof(hparams.w2v2_encoder_config__model_dim));
    fin.read((char*) &hparams.w2v2_encoder_config__max_seq_len, sizeof(hparams.w2v2_encoder_config__max_seq_len));
    fin.read((char*) &hparams.w2v2_encoder_config__feature_dim, sizeof(hparams.w2v2_encoder_config__feature_dim));
    fin.read((char*) &hparams.w2v2_encoder_config__use_fbank, sizeof(hparams.w2v2_encoder_config__use_fbank));
    fin.read((char*) &hparams.w2v2_encoder_config__first_pass_dropout_p, sizeof(hparams.w2v2_encoder_config__first_pass_dropout_p));
    fin.read((char*) &hparams.w2v2_encoder_config__layer_norm_features, sizeof(hparams.w2v2_encoder_config__layer_norm_features));
    fin.read((char*) &hparams.w2v2_encoder_config__feature_extractor_bias, sizeof(hparams.w2v2_encoder_config__feature_extractor_bias));
    fin.read((char*) &hparams.w2v2_encoder_config__feature_extractor_layer_norm_convs, sizeof(hparams.w2v2_encoder_config__feature_extractor_layer_norm_convs));
    fin.read((char*) &hparams.w2v2_encoder_config__feature_grad_scale, sizeof(hparams.w2v2_encoder_config__feature_grad_scale));
    fin.read((char*) &hparams.w2v2_encoder_config__num_fbank_channels, sizeof(hparams.w2v2_encoder_config__num_fbank_channels));
    fin.read((char*) &hparams.w2v2_encoder_config__fbank_stride, sizeof(hparams.w2v2_encoder_config__fbank_stride));
    fin.read((char*) &hparams.w2v2_encoder_config__sample_fbank_every_k, sizeof(hparams.w2v2_encoder_config__sample_fbank_every_k));
    fin.read((char*) &hparams.w2v2_encoder_config__pos_encoder_depth, sizeof(hparams.w2v2_encoder_config__pos_encoder_depth));
    fin.read((char*) &hparams.w2v2_encoder_config__pos_conv_kernel_size, sizeof(hparams.w2v2_encoder_config__pos_conv_kernel_size));
    fin.read((char*) &hparams.w2v2_encoder_config__num_pos_conv_groups, sizeof(hparams.w2v2_encoder_config__num_pos_conv_groups));
    fin.read((char*) &hparams.w2v2_encoder_config__use_conformer, sizeof(hparams.w2v2_encoder_config__use_conformer));
    fin.read((char*) &hparams.w2v2_encoder_config__num_encoder_layers, sizeof(hparams.w2v2_encoder_config__num_encoder_layers));
    fin.read((char*) &hparams.w2v2_encoder_config__num_encoder_attn_heads, sizeof(hparams.w2v2_encoder_config__num_encoder_attn_heads));
    fin.read((char*) &hparams.w2v2_encoder_config__ffn_inner_dim, sizeof(hparams.w2v2_encoder_config__ffn_inner_dim));
    fin.read((char*) &hparams.w2v2_encoder_config__dropout_p, sizeof(hparams.w2v2_encoder_config__dropout_p));
    fin.read((char*) &hparams.w2v2_encoder_config__attn_dropout_p, sizeof(hparams.w2v2_encoder_config__attn_dropout_p));
    fin.read((char*) &hparams.w2v2_encoder_config__layer_drop_p, sizeof(hparams.w2v2_encoder_config__layer_drop_p));
    fin.read((char*) &hparams.w2v2_encoder_config__norm_order, sizeof(hparams.w2v2_encoder_config__norm_order));
    fin.read((char*) &hparams.w2v2_encoder_config__depthwise_conv_kernel_size, sizeof(hparams.w2v2_encoder_config__depthwise_conv_kernel_size));
    fin.read((char*) &hparams.nllb_config__model_dim, sizeof(hparams.nllb_config__model_dim));
    fin.read((char*) &hparams.nllb_config__max_seq_len, sizeof(hparams.nllb_config__max_seq_len));
    fin.read((char*) &hparams.nllb_config__vocabulary_size, sizeof(hparams.nllb_config__vocabulary_size));
    fin.read((char*) &hparams.nllb_config__pad_idx, sizeof(hparams.nllb_config__pad_idx));
    fin.read((char*) &hparams.nllb_config__num_encoder_layers, sizeof(hparams.nllb_config__num_encoder_layers));
    fin.read((char*) &hparams.nllb_config__num_decoder_layers, sizeof(hparams.nllb_config__num_decoder_layers));
    fin.read((char*) &hparams.nllb_config__num_encoder_attn_heads, sizeof(hparams.nllb_config__num_encoder_attn_heads));
    fin.read((char*) &hparams.nllb_config__num_decoder_attn_heads, sizeof(hparams.nllb_config__num_decoder_attn_heads));
    fin.read((char*) &hparams.nllb_config__ffn_inner_dim, sizeof(hparams.nllb_config__ffn_inner_dim));
    fin.read((char*) &hparams.nllb_config__dropout_p, sizeof(hparams.nllb_config__dropout_p));
    fin.read((char*) &hparams.t2u_config__model_dim, sizeof(hparams.t2u_config__model_dim));
    fin.read((char*) &hparams.t2u_config__unit_max_seq_len, sizeof(hparams.t2u_config__unit_max_seq_len));
    fin.read((char*) &hparams.t2u_config__unit_vocabulary_size, sizeof(hparams.t2u_config__unit_vocabulary_size));
    fin.read((char*) &hparams.t2u_config__unit_pad_idx, sizeof(hparams.t2u_config__unit_pad_idx));
    fin.read((char*) &hparams.t2u_config__num_encoder_layers, sizeof(hparams.t2u_config__num_encoder_layers));
    fin.read((char*) &hparams.t2u_config__num_decoder_layers, sizeof(hparams.t2u_config__num_decoder_layers));
    fin.read((char*) &hparams.t2u_config__num_encoder_attn_heads, sizeof(hparams.t2u_config__num_encoder_attn_heads));
    fin.read((char*) &hparams.t2u_config__num_decoder_attn_heads, sizeof(hparams.t2u_config__num_decoder_attn_heads));
    fin.read((char*) &hparams.t2u_config__ffn_inner_dim, sizeof(hparams.t2u_config__ffn_inner_dim));
    fin.read((char*) &hparams.t2u_config__dropout_p, sizeof(hparams.t2u_config__dropout_p));
    fin.read((char*) &hparams.use_text_encoder, sizeof(hparams.use_text_encoder));
    fin.read((char*) &hparams.use_conformer_adaptor, sizeof(hparams.use_conformer_adaptor));
    fin.read((char*) &hparams.num_adaptor_layers, sizeof(hparams.num_adaptor_layers));
    fin.read((char*) &hparams.adaptor_kernel_size, sizeof(hparams.adaptor_kernel_size));
    fin.read((char*) &hparams.adaptor_stride, sizeof(hparams.adaptor_stride));
    fin.read((char*) &hparams.adaptor_layer_norm, sizeof(hparams.adaptor_layer_norm));
    fin.read((char*) &hparams.adaptor_dropout_p, sizeof(hparams.adaptor_dropout_p));
};

std::size_t
unity_model_loader::compute_context_size(unity_hparams &hparams)
{
    // TODO
};

void
unity_model_loader::init_model_tensors(fairseq2_model<unity_hparams> &model)
{
    // TODO
};
