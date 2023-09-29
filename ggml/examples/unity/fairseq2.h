#include <map>
#include <string>
#include "ggml.h"


struct fairseq2_model {
    ggml_context* ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
    void* hparams;
};

fairseq2_model fairseq2_model_alloc(ggml_context* ctx, void* hparams);

struct Linear {
    struct ggml_tensor* weight;  // out_dim * in_dim
    struct ggml_tensor* bias;  // out_dim
};

std::size_t Linear_size(int32_t input_dim, int32_t output_dim);
void Linear_init(Linear* self,fairseq2_model& model, const std::string &prefix, int input_dim, int output_dim, bool bias);

// LayerNorm

struct LayerNorm {
    struct ggml_tensor* weight;  // model_dim
    struct ggml_tensor* bias;  // model_dim
};

std::size_t LayerNorm_size(int32_t dim);

void LayerNorm_init(LayerNorm* self, fairseq2_model& model, const std::string &prefix, int dim);

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

struct StandardFeedForwardNetwork {
    struct Linear inner_proj; // ffn_inner_dim x model_dim
    // inner_activation -> Relu for unity
    // struct Dropout inner_dropout;
    struct LayerNorm inner_layer_norm; // ffn_inner_dim
    struct Linear output_proj; // model_dim x ffn_inner_dim
};

std::size_t StandardFeedForwardNetwork_size(int32_t dim, int32_t inner_dim);

void StandardFeedForwardNetwork_init(
    StandardFeedForwardNetwork* self,
    fairseq2_model& model,
    const std::string &prefix,
    int model_dim,
    int inner_dim
);

ggml_tensor* StandardFeedForwardNetwork_forward(
    StandardFeedForwardNetwork* self,
    ggml_tensor* seqs
);

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
