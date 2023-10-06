#include <string>
#include "model_loader.h"

#define DEBUG_MODEL_LOAD 0

std::ifstream open_ggml_file(const char* fname) {
    printf("%s: loading model from '%s'\n", __func__, fname);

    auto fin = std::ifstream(std::string(fname), std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }

    std::uint32_t magic;
    fin.read((char*)&magic, 4);
    if (magic != GGML_FILE_MAGIC) {
        fprintf(stderr, "%s: invalid model file '%s' (bad header %d)\n", __func__, fname, magic);
        throw std::invalid_argument("failed to open file."); // TODO Merge error message.
    }
    return fin;
}

int
model_loader::load_model_weights(fairseq2_model &model, std::ifstream &fin)
{
    size_t total_size = 0;
    while (!fin.eof()) {
        std::string name = get_name(fin);
        if (name.length() == 0)
            break;
        auto tensor = load_tensor_value(fin, model.tensors_ctx);
        if (tensor == nullptr) {
            // Abort in case of error, the input stream is corrupted at this point.
            printf("Error while reading tensor %s\n", name.c_str() );
            return 1;
        }
        model.tensors[name] = tensor;
        if (DEBUG_MODEL_LOAD) {
            printf("%s [%5ld, %5ld], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), tensor->ne[0], tensor->ne[1], ggml_type_name(tensor->type), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
        }
        total_size += ggml_nbytes(tensor);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    return 0;
};

ggml_tensor* load_tensor_value(std::ifstream &fin, ggml_context* ctx)
{
    int32_t n_dims = 0;
    int32_t raw_type = 0;

    fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
    fin.read(reinterpret_cast<char *>(&raw_type),  sizeof(raw_type));
    ggml_type type = ggml_type(raw_type);

    if (n_dims <= 0 || n_dims > GGML_MAX_DIMS || raw_type < 0 || raw_type > GGML_TYPE_COUNT) {
        return nullptr;
    }
    int64_t ne[4] = {1, 1, 1, 1};
    for (int i = 0; i < n_dims; ++i) {
        fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
    }

    ggml_tensor* tensor = ggml_new_tensor(ctx, type, n_dims, ne);
    fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
    return tensor;
};

std::string
model_loader::get_name(std::ifstream& fin)
{
    std::uint32_t length = 0;
    fin.read(reinterpret_cast<char *>(&length), sizeof(length));
    std::string name(length, 0);
    if (length == 0) {
        return name;
    };
    fin.read(&name[0], length);

    return name;
};
