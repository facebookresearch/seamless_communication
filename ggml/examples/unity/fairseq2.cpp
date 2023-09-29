#include "ggml.h"
#include "fairseq2.h"

fairseq2_model fairseq2_model_init(ggml_context* ctx, void* hparams) {
    // TODO? allocate the model in the ggml_context
    fairseq2_model model;
    model.ctx = ctx;
    model.hparams = hparams;
    // TODO:
    // init_model_tensors(model);
    return model;
};

// Linear

std::size_t Linear_size(int32_t input_dim, int32_t output_dim)
{
    return (input_dim * output_dim * ggml_type_size(GGML_TYPE_F32)) // weight
        + (output_dim * ggml_type_size(GGML_TYPE_F32)); // bias
};

void Linear_init(
    Linear* self,
    fairseq2_model& model,
    const std::string &prefix,
    int input_dim,
    int output_dim,
    bool bias
) {
    self->weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, output_dim, input_dim);
    model.tensors[prefix + ".weight"] = self->weight;
    if (bias) {
        self->bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, output_dim);
        model.tensors[prefix + ".inner_proj.bias"] = self->bias;
    }
}

// LayerNorm

std::size_t LayerNorm_size(int32_t dim)
{
    return 2 * dim * ggml_type_size(GGML_TYPE_F32); // weight and bias
};

void LayerNorm_init(
    LayerNorm* self,
    fairseq2_model& model,
    const std::string &prefix,
    int dim
) {
    self->weight = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, dim);
    model.tensors[prefix + ".weight"] = self->weight;
    self->bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, dim);
    model.tensors[prefix + ".bias"] = self->bias;
}

std::size_t StandardFeedForwardNetwork_size(int32_t dim, int32_t inner_dim)
{
    return LayerNorm_size(dim) + Linear_size(dim, inner_dim) + Linear_size(inner_dim, dim);
};

void StandardFeedForwardNetwork_init(
    StandardFeedForwardNetwork* self,
    fairseq2_model& model,
    const std::string &prefix,
    int model_dim,
    int inner_dim
) {
    Linear_init(&self->inner_proj, model, prefix + ".inner_proj", model_dim, inner_dim, true);
    LayerNorm_init(&self->inner_layer_norm, model, prefix + ".inner_layer_norm", inner_dim);
    Linear_init(&self->output_proj, model, prefix + ".output_proj", inner_dim, model_dim, true);
}

ggml_tensor* StandardFeedForwardNetwork_forward(
    StandardFeedForwardNetwork* self,
    ggml_tensor* seqs
) {
    return seqs;
}
