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

extern "C" ggml_tensor* StandardTransformerEncoderLayer_forward(
    fairseq2_model& model,
    const std::string& prefix,
    ggml_tensor* seqs,
    ggml_tensor* padding_mask
);

// Specifies the Layer Normalization order.
enum TransformerNormOrder {
    TRANSFORMER_NORM_ORDER_POST = 0,
    TRANSFORMER_NORM_ORDER_PRE = 1,
    TRANSFORMER_NORM_ORDER_PRE_WITH_NORMFORMER = 2
};



/// Holds the options to pass to a sequence generator.
struct SequenceGeneratorOptions {
    /// The beam size.
    int beam_size = 5;

    /// The minimum length of generated sequences (including prefix sequence).
    int min_seq_len = 1;

    /// The terms ``a`` and ``b`` of ``ax + b`` where ``x`` is the source
    /// sequence length. The generated sequences (including prefix sequence) will
    /// have the maximum length of ``min(hard_max_seq_len, ax + b)``. See also
    /// ``hard_max_seq_len``.
    int soft_max_seq_len_a = 1;
    int soft_max_seq_len_b = 200;

    /// The hard limit on maximum length of generated sequences.
    int hard_max_seq_len = 1024;

    /// The length penalty, where values less than 1.0 favor shorter, values
    /// greater than 1.0 favor longer sequences.
    float len_penalty = 1.0;

    /// The unknown symbol penalty, where values less than 0 produce more UNKs,
    /// values greater than 0 produce fewer UNKs.
    float unk_penalty = 0.0;

    /// If ``True``, normalizes scores by the length of generated sequences.
    bool normalize_scores = true;
};


struct SequenceGeneratorJob {
    SequenceGeneratorOptions opts;
    ggml_tensor* prefix_seq;
    int source_seq_len;
    std::int32_t eos_idx;
};


extern "C" float generate_sequence(
    fairseq2_model& model,
    const SequenceGeneratorJob& opts,
    ggml_tensor* encoder_output,
    ggml_tensor* encoder_padding_mask,
    ggml_tensor** output_seq
);
