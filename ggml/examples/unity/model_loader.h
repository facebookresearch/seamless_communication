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

    virtual void load_hparams(fairseq2_model& model, std::ifstream &fin) = 0;

    virtual std::size_t compute_context_size(void *raw_hparams) = 0;

    int load_model_weights(fairseq2_model &model, std::ifstream &fin);

private:
    ggml_tensor * next_tensor(std::ifstream &fin, fairseq2_model &model);

    std::string get_name(std::ifstream &fin);
};

ggml_tensor* load_tensor_value(std::ifstream &fin, ggml_context* ctx);

std::ifstream open_ggml_file(const char* fname);

template<typename T>
int load_fairseq2_ggml_file(fairseq2_model& model, const char* fname) {
    T loader;
    auto fin = open_ggml_file(fname);
    loader.load_hparams(model, fin);

    std::size_t ctx_size = loader.compute_context_size(model.hparams);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    model.tensors_ctx = ggml_init(params);

    return loader.load_model_weights(model, fin);
}

