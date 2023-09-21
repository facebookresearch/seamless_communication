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

#include <iostream>
#include <stdexcept>


template <typename T>
struct fairseq2_model {
    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;

    T hparams;
};

template <typename T>
class model_loader {
public:
    void
    load_ggml_file(const std::string &fname, fairseq2_model<T> &model);

protected:
    virtual void
    load_hparams(std::ifstream &fin, T &hparams) = 0;

    virtual std::size_t
    compute_context_size(T &hparams) = 0;

    virtual void
    init_model_tensors(fairseq2_model<T> &model);

private:
    bool verify_magic(std::ifstream &fin);

    void
    init_model(fairseq2_model<T> &model);

    void load_model_weights(std::ifstream &fin, fairseq2_model<T> &model);
    
    ggml_tensor * next_tensor(std::ifstream &fin, fairseq2_model<T> &model);

    // TODO Move these two to helpers
    void load_tensor_value(std::ifstream &fin, ggml_tensor *tensor);
    std::string get_name(std::ifstream &fin);
};