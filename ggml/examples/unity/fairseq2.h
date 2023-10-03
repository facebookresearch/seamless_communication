#pragma once

#include <map>
#include <string>
#include <vector>
#include "ggml.h"


struct fairseq2_model {
    ggml_context* ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
    void* arch;
    void* hparams;
};

/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc();
extern "C" void fairseq2_model_free(fairseq2_model* model);

extern "C" std::string* std_string_alloc(char* c_str);
extern "C" void std_string_free(std::string* str);


struct Linear {
    struct ggml_tensor* weight;  // out_dim * in_dim
    struct ggml_tensor* bias;  // out_dim
};

std::size_t Linear_size(int32_t input_dim, int32_t output_dim);
void Linear_init(Linear& self,fairseq2_model& model, const std::string &prefix, int input_dim, int output_dim, bool bias);

// LayerNorm

struct LayerNorm {
    struct ggml_tensor* weight;  // model_dim
    struct ggml_tensor* bias;  // model_dim
};

std::size_t LayerNorm_size(int32_t dim);

void LayerNorm_init(LayerNorm& self, fairseq2_model& model, const std::string &prefix, int dim);

// ConformerConvolution
// struct ConformerConvolution {
//     // pointwise_conv1: Conv1d
//     // pointwise_conv1_activation: GLU
//     // depthwise_conv: Conv1d
//     // batch_norm: BatchNorm1d
//     // depthwise_activation: Module
//     // pointwise_conv2: Conv1d
// };

// std::size_t ConformerConvolution_size(int32_t dim);

// void ConformerConvolution_init(ConformerConvolution* self, fairseq2_model& model, const std::string &prefix, int dim);



struct MultiheadAttention {
    // num_key_value_heads: int
    struct Linear q_proj;
    struct Linear k_proj;
    struct Linear v_proj;
    // pos_encoder: Optional[PositionEncoder]
    struct ggml_tensor* bias_k;
    struct ggml_tensor* bias_v;
    // add_zero_attn: bool
    // head_scale_weight: Optional[Parameter]
    struct Linear output_proj;
};

void MultiheadAttention_init(MultiheadAttention& self, fairseq2_model& model, const std::string &prefix, int model_dim, int num_heads);

struct StandardFeedForwardNetwork {
    struct Linear inner_proj; // ffn_inner_dim x model_dim
    // inner_activation -> Relu for unity
    // struct Dropout inner_dropout;
    struct LayerNorm inner_layer_norm; // ffn_inner_dim
    struct Linear output_proj; // model_dim x ffn_inner_dim
};

std::size_t StandardFeedForwardNetwork_size(int32_t dim, int32_t inner_dim);

void StandardFeedForwardNetwork_init(
    StandardFeedForwardNetwork& self,
    fairseq2_model& model,
    const std::string &prefix,
    int model_dim,
    int inner_dim
);

extern "C" ggml_tensor* StandardFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* input
);

// Transformer

enum TransformerNormOrder {
    TRANSFORMER_NORM_ORDER_POST = 0,
    TRANSFORMER_NORM_ORDER_PRE = 1,
    TRANSFORMER_NORM_ORDER_PRE_WITH_NORMFORMER = 2
};


struct TransformerDecoderLayer {
    struct MultiheadAttention self_attn;
    struct LayerNorm self_attn_norm;
    // self_attn_dropout: Optional[Dropout]
    struct LayerNorm self_attn_layer_norm;
    struct MultiheadAttention encoder_decoder_attn;
    // encoder_decoder_dropout: Optional[Dropout]
    struct LayerNorm encoder_decoder_attn_layer_norm;
    struct StandardFeedForwardNetwork ffn;
    // ffn_dropout: Optional[Dropout]
    // residual_scale: Optional[Parameter]
    struct LayerNorm ffn_layer_norm;
    // norm_order: TransformerNormOrder
};

void TransformerDecoderLayer_init();


struct TransformerDecoder {
    std::vector<TransformerDecoderLayer> layers;
    struct LayerNorm layer_norm;
};

// std::size_t TransformerDecoder_size(int32_t input_dim, int32_t output_dim);
// void TransformerDecoder_init(TransformerEncoder* self, fairseq2_model& model, const std::string &prefix, TransformerNormOrder norm_order);


// std::size_t TransformerEncoder_size(int32_t input_dim, int32_t output_dim);
// void TransformerEncoder_init(TransformerEncoder* self, fairseq2_model& model, const std::string &prefix, TransformerNormOrder norm_order);

//
