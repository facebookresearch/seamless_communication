// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.


#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "common.h"
#include "common-ggml.h"

#include <iostream>
#include <stdexcept>

#include "ggml/examples/unity/model_loader.h"


template<typename T>
void
model_loader<T>::load_ggml_file(const std::string &fname, fairseq2_model<T> &model)
{
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }

    if (!verify_magic(fin)) {
        fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }

    load_hparams(fin, model.hparams);
    init_model(model);
    load_model_weights(fin, model);
};

template<typename T>
bool 
model_loader<T>::verify_magic(std::ifstream &fin)
{
    uint32_t magic;
    fin.read((char *) &magic, sizeof(magic));

    return magic == GGML_FILE_MAGIC;
};

template<typename T>
void
model_loader<T>::init_model(fairseq2_model<T> &model)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/ compute_context_size(model.hparams),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    model.ctx = ggml_init(params);
    if (!model.ctx)
        throw std::runtime_error("ggml_init() failed.");

    init_model_tensors(model);
};

template<typename T>
void
model_loader<T>::load_model_weights(std::ifstream &fin, fairseq2_model<T> &model)
{
    size_t total_size = 0;
    while (!fin.eof()) {
        auto tensor = next_tensor(fin, model);
        load_tensor_value(fin, tensor);
        total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
};

template<typename T>
ggml_tensor *
model_loader<T>::next_tensor(std::ifstream &fin, fairseq2_model<T> &model)
{
    auto name = get_name(fin);
    std::cout << "loading tensor: " << name << std::endl;
   
    if (model.tensors.find(name) == model.tensors.end()) {
        fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }

    return model.tensors[name];
};

template<typename T>
void
model_loader<T>::load_tensor_value(std::ifstream &fin, ggml_tensor *tensor)
{
    int32_t n_dims;
    int32_t ttype;

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

    int32_t nelements = 1;
    int32_t ne[3] = {1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        nelements *= ne[i];
    }

    if (ggml_nelements(tensor) != nelements) {
        std::cout << ggml_nelements(tensor) << std::endl;
        std::cout << nelements << std::endl;
        throw std::runtime_error("tensor has wrong size in model file.");
    }

    if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
        fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                __func__, (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
        throw std::runtime_error("tensor has wrong shape in file."); // TODO Merge error message.
    }

    // for debugging
    if (0) {
        printf("%[%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
    }

    const size_t bpe = ggml_type_size(ggml_type(ttype));

    if ((nelements * bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
        fprintf(stderr, "%s: tensor has wrong size in model file: got %zu, expected %zu\n",
                __func__, ggml_nbytes(tensor), nelements * bpe);
        throw std::runtime_error("tensor has wrong size in file."); // TODO Merge error message.
    }

    fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
};


template<typename T>
std::string
model_loader<T>::get_name(std::ifstream& fin)
{
    int32_t length;
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    std::string name(length, 0);
    fin.read(&name[0], length);
 
    return name;
};
