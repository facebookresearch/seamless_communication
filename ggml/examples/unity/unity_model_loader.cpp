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

void unity_model_loader::load_hparams(fairseq2_model& model, std::ifstream &fin)
{
    unity_hparams* hparams = (unity_hparams*)model.hparams;
    read_unity_hparams(hparams, fin);
    if (hparams->__end_of_hparams__ != 6877961321223123048) {
        throw std::invalid_argument("");
    }
}

std::size_t
unity_model_loader::compute_context_size(void* raw_hparams)
{
    auto* hparams = (unity_hparams*)raw_hparams;
    return hparams->model_byte_size;
};

extern "C" int load_unity_ggml_file(fairseq2_model& model, const char* fname) {
    return load_fairseq2_ggml_file<unity_model_loader>(model, fname);
}



