// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "common.h"
#include "common-ggml.h"
#include "fairseq2.h"

#include <iostream>
#include <stdexcept>

class model_loader {
public:
    virtual ~model_loader() {};

    virtual fairseq2_model& alloc_model(ggml_context* ctx) = 0;

    virtual void load_hparams(fairseq2_model& model, std::ifstream &fin) = 0;

    virtual void load_model_weights(fairseq2_model &model, std::ifstream &fin);

    virtual std::size_t
    compute_context_size(void *raw_hparams) = 0;

    virtual void
    init_model_tensors(fairseq2_model &model) = 0;

private:
    ggml_tensor * next_tensor(std::ifstream &fin, fairseq2_model &model);

    // TODO Move these two to helpers
    void load_tensor_value(std::ifstream &fin, ggml_tensor *tensor);
    std::string get_name(std::ifstream &fin);
};

/// allocate the fairseq2 model and hyperparameters into the ggml context
template<typename T>
fairseq2_model& alloc_fairseq2_model(ggml_context* ctx) {
    auto hparams = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(T))->data;
    auto& model = (fairseq2_model&)ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(fairseq2_model))->data;

    model.ctx = ctx;
    model.hparams = hparams;
    return model;
};

std::ifstream open_ggml_file(const char* fname);

template<typename T>
fairseq2_model& load_fairseq2_ggml_file(ggml_context* ctx, const char* fname) {
    T loader;
    fairseq2_model& model = loader.alloc_model(ctx);
    auto fin = open_ggml_file(fname);
    loader.load_hparams(model, fin);
    loader.load_model_weights(model, fin);
    return model;
}
