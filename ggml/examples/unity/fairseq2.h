#pragma once

#include <map>
#include <string>
#include <vector>
#include "ggml.h"


struct fairseq2_model {
    // Context containing all tensors memory
    ggml_context* tensors_ctx;
    // Named tensors, all tensors should belong to tensors_ctx
    std::map<std::string, struct ggml_tensor *> tensors;
    void* arch;
    void* hparams;
    // an inference context, not managed by this object
    // TODO: is this the best place to store this or should we also pass this to all forward methods ?
    ggml_context* ctx;
};

/// allocate the fairseq2 model and hyperparameters
extern "C" fairseq2_model* fairseq2_model_alloc();
// free the models and all its owned tensors
extern "C" void fairseq2_model_free(fairseq2_model* model);
extern "C" void fairseq2_model_set_inference_ctx(fairseq2_model* model, ggml_context* ctx);

extern "C" std::string* std_string_alloc(char* c_str);
extern "C" void std_string_free(std::string* str);


extern "C" ggml_tensor* Linear_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input
);

extern "C" ggml_tensor* LayerNorm_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* input
);

extern "C" ggml_tensor* StandardFeedForwardNetwork_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs
);

extern "C" ggml_tensor* MultiheadAttention_forward(
    fairseq2_model& model,
    const std::string &prefix,
    ggml_tensor* queries,  // (slen, d_in)
    ggml_tensor* keys,  // (klen, d_in)
    ggml_tensor* values,  // (klen, d_out)
    ggml_tensor* _ // (klen, slen)  TODO: do we need to pass mask here ?
);
