#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "common.h"
#include "common-ggml.h"
#include "math.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>

// default hparams
struct unity_hparams {
    int32_t n_text_vocab = 256206; 
    int32_t n_unit_vocab = 10084; 
    int32_t n_audio_enc_dim = 1024;
    int32_t n_audio_enc_ffn_dim = 4096;
    int32_t n_audio_enc_feat_dim = 160;
    int32_t n_audio_enc_layer = 24;
    int32_t n_audio_enc_head = 16;
    int32_t ftype   = 1;
    float   eps     = 1e-5f;
};
// layer def
struct audio_enc_layer {
    struct ggml_tensor * self_attn_layer_norm_w;
    struct ggml_tensor * self_attn_layer_norm_b;

    struct ggml_tensor * self_attn_linear_k_w;
    struct ggml_tensor * self_attn_linear_k_b;
    struct ggml_tensor * self_attn_linear_q_w;
    struct ggml_tensor * self_attn_linear_q_b;
    struct ggml_tensor * self_attn_linear_v_w;
    struct ggml_tensor * self_attn_linear_v_b;
    struct ggml_tensor * self_attn_linear_out_w;
    struct ggml_tensor * self_attn_linear_out_b;
    struct ggml_tensor * self_attn_linear_pos_w;

    struct ggml_tensor * self_attn_pos_bias_u;
    struct ggml_tensor * self_attn_pos_bias_v;

    struct ggml_tensor * conv_layer_norm_w;
    struct ggml_tensor * conv_layer_norm_b;

    struct ggml_tensor * conv_pointwise_conv1_w;
    struct ggml_tensor * conv_depthwise_conv_w;
    struct ggml_tensor * conv_batch_norm_w;
    struct ggml_tensor * conv_batch_norm_b;
    struct ggml_tensor * conv_batch_norm_running_mean;
    struct ggml_tensor * conv_batch_norm_running_var;
    struct ggml_tensor * conv_batch_norm_num_batches_tracked;
    struct ggml_tensor * conv_pointwise_conv2_w;

    struct ggml_tensor * ffn1_layer_norm_w;
    struct ggml_tensor * ffn1_layer_norm_b;
    struct ggml_tensor * ffn1_w1;
    struct ggml_tensor * ffn1_b1;
    struct ggml_tensor * ffn1_w2;
    struct ggml_tensor * ffn1_b2;

    struct ggml_tensor * ffn2_layer_norm_w;
    struct ggml_tensor * ffn2_layer_norm_b;
    struct ggml_tensor * ffn2_w1;
    struct ggml_tensor * ffn2_b1;
    struct ggml_tensor * ffn2_w2;
    struct ggml_tensor * ffn2_b2;

    struct ggml_tensor * final_layer_norm_w;
    struct ggml_tensor * final_layer_norm_b;
};

// struct ggml_tensor * conv_ln;
// struct ggml_tensor * conv_pool_1d;

// model def
struct unity_model {
    unity_hparams hparams;
    // audio encoder
    struct ggml_tensor * post_extract_proj_w;
    struct ggml_tensor * post_extract_proj_b;
    struct ggml_tensor * audio_enc_pos_conv_wg;
    struct ggml_tensor * audio_enc_pos_conv_wv;
    struct ggml_tensor * audio_enc_pos_conv_b;
    struct ggml_tensor * audio_enc_layer_norm_w;
    struct ggml_tensor * audio_enc_layer_norm_b;
    struct ggml_tensor * audio_enc_pos_enc_w;
    struct ggml_tensor * layer_norm_w;
    struct ggml_tensor * layer_norm_b;
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;
    std::vector<audio_enc_layer> audio_enc_layers;

    // text encoder
    // std::vector<text_enc_layer> text_enc_layers;

    // adaptor
    // std::vector<adapter_layer> adapter_layers;

    // text decoder
    // std::vector<text_dec_layer> text_dec_layers;

    // unit decoder
    // std::vector<unit_dec_layer> unit_dec_layers;

    //
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// model load
bool unity_model_load(const std::string & fname, unity_model & model, gpt_vocab & vocab) {
    printf("%s: loading model from '%s'\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != GGML_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto & hparams = model.hparams;

        fin.read((char *) &hparams.n_text_vocab, sizeof(hparams.n_text_vocab));
        fin.read((char *) &hparams.n_audio_enc_dim,  sizeof(hparams.n_audio_enc_dim));
        fin.read((char *) &hparams.n_audio_enc_ffn_dim,  sizeof(hparams.n_audio_enc_ffn_dim));
        fin.read((char *) &hparams.n_audio_enc_feat_dim,  sizeof(hparams.n_audio_enc_feat_dim));
        fin.read((char *) &hparams.n_audio_enc_layer, sizeof(hparams.n_audio_enc_layer));
        fin.read((char *) &hparams.n_audio_enc_head, sizeof(hparams.n_audio_enc_head));
        fin.read((char *) &hparams.ftype,   sizeof(hparams.ftype));

        const int32_t qntvr = hparams.ftype / GGML_QNT_VERSION_FACTOR;

        printf("%s: n_text_vocab = %d\n", __func__, hparams.n_text_vocab);
        printf("%s: n_audio_enc_dim   = %d\n", __func__, hparams.n_audio_enc_dim);
        printf("%s: n_audio_enc_ffn_dim  = %d\n", __func__, hparams.n_audio_enc_ffn_dim);
        printf("%s: n_audio_enc_feat_dim = %d\n", __func__, hparams.n_audio_enc_feat_dim);
        printf("%s: n_audio_enc_layer = %d\n", __func__, hparams.n_audio_enc_layer);
        printf("%s: n_audio_enc_head = %d\n", __func__, hparams.n_audio_enc_head);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
        printf("%s: qntvr   = %d\n", __func__, qntvr);

        hparams.ftype %= GGML_QNT_VERSION_FACTOR;
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto & hparams = model.hparams;

        const int n_audio_enc_dim  = hparams.n_audio_enc_dim;
        const int n_audio_enc_ffn_dim  = hparams.n_audio_enc_ffn_dim;
        const int n_audio_enc_layer = hparams.n_audio_enc_layer;
        const int n_ctx = 4096;  // 20ms * 4096 = 80s
        // const int n_text_vocab = hparams.n_text_vocab;
        const int kernel_size = 31;

        ctx_size += n_audio_enc_layer*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // self_attn_layer_norm_w
        ctx_size += n_audio_enc_layer*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // self_attn_layer_norm_b

        ctx_size += n_audio_enc_layer*(5*n_audio_enc_dim*n_audio_enc_dim*ggml_type_sizef(wtype));         // self_attn_w
        ctx_size += n_audio_enc_layer*(4*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // self_attn_b

        ctx_size += n_audio_enc_layer*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // conv_layer_norm_w
        ctx_size += n_audio_enc_layer*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // conv_layer_norm_b

        ctx_size += n_audio_enc_layer*(n_audio_enc_dim*n_audio_enc_dim*2*ggml_type_sizef(wtype));           // conv_pointwise_conv1_w
        ctx_size += n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // conv_batch_norm_w
        ctx_size += n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // conv_batch_norm_b
        ctx_size += n_audio_enc_layer*(n_audio_enc_dim*n_audio_enc_dim*kernel_size*ggml_type_sizef(wtype));         // conv_depthwise_conv_w
        ctx_size += n_audio_enc_layer*(n_audio_enc_dim*n_audio_enc_dim*ggml_type_sizef(wtype));           // conv_pointwise_conv2_w

        ctx_size += 2 * n_audio_enc_layer * (n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // ffn{1,2}_layer_norm_w
        ctx_size += 2 * n_audio_enc_layer * (n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // ffn{1,2}_layer_norm_b
        ctx_size += 2 * n_audio_enc_layer * (2 * n_audio_enc_dim * n_audio_enc_ffn_dim * ggml_type_sizef(wtype));  // ffn{1,2}_w{1,2}
        ctx_size += 2 * n_audio_enc_layer * (2 * n_audio_enc_dim * ggml_type_sizef(GGML_TYPE_F32));  // ffn{1,2}_b{1,2}

        ctx_size += n_audio_enc_layer*(n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // final_layer_norm_w
        ctx_size += n_audio_enc_layer*(n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // final_layer_norm_b

        ctx_size += n_ctx*n_audio_enc_layer*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // memory_k
        ctx_size += n_ctx*n_audio_enc_layer*n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32); // memory_v

        // Adaptor
        // ctx_size += n_audio_enc_layer*(n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // conv_ln
        // ctx_size += n_audio_enc_layer*(n_audio_enc_dim*ggml_type_sizef(GGML_TYPE_F32)); // conv_pool_1d

        // object overhead might differ depending on the structure and other miscellaneous factors
        ctx_size += (6 + 12*n_audio_enc_layer)*512; // updated object overhead

        printf("%s: ggml tensor size = %d bytes\n", __func__, (int) sizeof(ggml_tensor));
        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_audio_enc_dim  = hparams.n_audio_enc_dim;
        const int n_audio_enc_ffn_dim  = hparams.n_audio_enc_ffn_dim;
        const int n_audio_enc_feat_dim = hparams.n_audio_enc_feat_dim;
        const int n_audio_enc_layer = hparams.n_audio_enc_layer;
        const int n_audio_enc_head = hparams.n_audio_enc_head;
        const int n_ctx = 4096;  // 20ms * 4096 = 80s
        const int pos_conv_kernel_size = 128;
        // const int n_text_vocab = hparams.n_text_vocab;

        model.audio_enc_layers.resize(n_audio_enc_layer);
        model.audio_enc_pos_enc_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_ctx * 2 - 1);
        model.tensors["model/enc/pos_enc/w"] = model.audio_enc_pos_enc_w;

        model.post_extract_proj_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_feat_dim, n_audio_enc_dim);
        model.post_extract_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
        model.tensors["model/post_extract_proj/w"] = model.post_extract_proj_w;
        model.tensors["model/post_extract_proj/b"] = model.post_extract_proj_b;

        model.audio_enc_pos_conv_wg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, pos_conv_kernel_size);
        model.audio_enc_pos_conv_wv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, pos_conv_kernel_size, 64, n_audio_enc_dim);
        model.audio_enc_pos_conv_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
        model.tensors["model/enc/pos_conv/w_g"] = model.audio_enc_pos_conv_wg;
        model.tensors["model/enc/pos_conv/w_v"] = model.audio_enc_pos_conv_wv;
        model.tensors["model/enc/pos_conv/b"] = model.audio_enc_pos_conv_b;

        model.audio_enc_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
        model.audio_enc_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
        model.tensors["model/enc/layer_norm/w"] = model.audio_enc_layer_norm_w;
        model.tensors["model/enc/layer_norm/b"] = model.audio_enc_layer_norm_b;

        model.layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_feat_dim);
        model.layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_feat_dim);
        model.tensors["model/layer_norm/w"] = model.layer_norm_w;
        model.tensors["model/layer_norm/b"] = model.layer_norm_b;



        for (int i = 0; i < n_audio_enc_layer; ++i) {
            auto & layer = model.audio_enc_layers[i];

            layer.self_attn_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.self_attn_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            layer.self_attn_linear_k_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_dim);
            layer.self_attn_linear_k_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.self_attn_linear_q_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_dim);
            layer.self_attn_linear_q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.self_attn_linear_v_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_dim);
            layer.self_attn_linear_v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.self_attn_linear_out_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_dim);
            layer.self_attn_linear_out_b   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.self_attn_linear_pos_w   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_dim);

            layer.self_attn_pos_bias_u = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim / n_audio_enc_head, n_audio_enc_head);
            layer.self_attn_pos_bias_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim / n_audio_enc_head, n_audio_enc_head);

            layer.conv_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.conv_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            layer.conv_pointwise_conv1_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, 2*n_audio_enc_dim);
            layer.conv_depthwise_conv_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 31, n_audio_enc_dim);

            layer.conv_batch_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.conv_batch_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.conv_batch_norm_running_mean = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.conv_batch_norm_running_var = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.conv_batch_norm_num_batches_tracked = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

            layer.conv_pointwise_conv2_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_dim);

            layer.ffn1_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.ffn1_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            layer.ffn1_w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_ffn_dim);
            layer.ffn1_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_ffn_dim);

            layer.ffn1_w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_ffn_dim, n_audio_enc_dim);
            layer.ffn1_b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            layer.ffn2_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.ffn2_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            layer.ffn2_w1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_dim, n_audio_enc_ffn_dim);
            layer.ffn2_b1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_ffn_dim);

            layer.ffn2_w2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_audio_enc_ffn_dim, n_audio_enc_dim);
            layer.ffn2_b2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            layer.final_layer_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);
            layer.final_layer_norm_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_audio_enc_dim);

            // map by name

            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_layer_norm/w"] = layer.self_attn_layer_norm_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_layer_norm/b"] = layer.self_attn_layer_norm_b;

            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_k/w"] = layer.self_attn_linear_k_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_k/b"] = layer.self_attn_linear_k_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_q/w"] = layer.self_attn_linear_q_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_q/b"] = layer.self_attn_linear_q_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_v/w"] = layer.self_attn_linear_v_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_v/b"] = layer.self_attn_linear_v_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_out/w"] = layer.self_attn_linear_out_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_out/b"] = layer.self_attn_linear_out_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_linear_pos/w"] = layer.self_attn_linear_pos_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_pos_bias/u"] = layer.self_attn_pos_bias_u;
            model.tensors["model/enc/h" + std::to_string(i) + "/self_attn_pos_bias/v"] = layer.self_attn_pos_bias_v;

            model.tensors["model/enc/h" + std::to_string(i) + "/conv_layer_norm/w"]        = layer.conv_layer_norm_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_layer_norm/b"]        = layer.conv_layer_norm_b;

            model.tensors["model/enc/h" + std::to_string(i) + "/conv_pointwise_conv1/w"] = layer.conv_pointwise_conv1_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_depthwise_conv/w"] = layer.conv_depthwise_conv_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_batch_norm/w"] = layer.conv_batch_norm_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_batch_norm/b"] = layer.conv_batch_norm_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_batch_norm/m"] = layer.conv_batch_norm_running_mean;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_batch_norm/v"] = layer.conv_batch_norm_running_var;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_batch_norm/n"] = layer.conv_batch_norm_num_batches_tracked;
            model.tensors["model/enc/h" + std::to_string(i) + "/conv_pointwise_conv2/w"] = layer.conv_pointwise_conv2_w;

            model.tensors["model/enc/h" + std::to_string(i) + "/ffn1_layer_norm/w"] = layer.ffn1_layer_norm_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn1_layer_norm/b"] = layer.ffn1_layer_norm_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn1_w_1/w"] = layer.ffn1_w1;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn1_w_1/b"] = layer.ffn1_b1;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn1_w_2/w"] = layer.ffn1_w2;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn1_w_2/b"] = layer.ffn1_b2;

            model.tensors["model/enc/h" + std::to_string(i) + "/ffn2_layer_norm/w"] = layer.ffn2_layer_norm_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn2_layer_norm/b"] = layer.ffn2_layer_norm_b;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn2_w_1/w"] = layer.ffn2_w1;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn2_w_1/b"] = layer.ffn2_b1;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn2_w_2/w"] = layer.ffn2_w2;
            model.tensors["model/enc/h" + std::to_string(i) + "/ffn2_w_2/b"] = layer.ffn2_b2;

            model.tensors["model/enc/h" + std::to_string(i) + "/final_layer_norm/w"] = layer.final_layer_norm_w;
            model.tensors["model/enc/h" + std::to_string(i) + "/final_layer_norm/b"] = layer.final_layer_norm_b;
        }
    }


    // load weights
    {
        size_t total_size = 0;
        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[3] = { 1, 1, 1};
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            std::cout << "loading " << name << " " << n_dims << std::endl;

            if (model.tensors.find(name) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.c_str());
                return false;
            }

            auto tensor = model.tensors[name];
            if (ggml_nelements(tensor) != nelements) {
                std::cout << ggml_nelements(tensor) << std::endl;
                std::cout << nelements << std::endl;
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.c_str());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.c_str(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.c_str(), ne[0], ne[1], ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor)/1024.0/1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.c_str(), ggml_nbytes(tensor), nelements*bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
            total_size += ggml_nbytes(tensor);
        }

        printf("%s: model size  = %8.2f MB\n", __func__, total_size/1024.0/1024.0);
    }

    fin.close();

    return true;
}

// build the computation graph
struct ggml_cgraph * unity_graph(
        const unity_model & model,
        struct ggml_allocr * allocr) {

    const auto & hparams = model.hparams;

    const int n_audio_enc_dim  = hparams.n_audio_enc_dim;
    const int n_audio_enc_ffn_dim  = hparams.n_audio_enc_ffn_dim;
    const int n_audio_enc_feat_dim = hparams.n_audio_enc_feat_dim;
    const int n_audio_enc_layer = hparams.n_audio_enc_layer;
    const int n_audio_enc_head = hparams.n_audio_enc_head;
    const int n_ctx = 4096;  // 20ms * 4096 = 80s
    const int pos_conv_kernel_size = 128;
    // const int n_text_vocab = hparams.n_text_vocab;
    const int kernel_size = 31;

    // since we are using ggml-alloc, this buffer only needs enough space to hold the ggml_tensor and ggml_cgraph structs, but not the tensor data
    static size_t buf_size = ggml_tensor_overhead()*GGML_MAX_NODES + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    struct ggml_context * ctx0 = ggml_init(params);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    /// For dev, load an example input before conformer blocks
    auto file = std::ifstream("/private/home/dnn/internal_sc/seamless_communication/ggml/examples/unity/dev/seqs_before_conformer_block.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open binary file." << std::endl;
    }
    struct ggml_tensor * inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1024, 137);
    inpL->data = malloc(ggml_nbytes(inpL));
    file.read(reinterpret_cast<char *>(inpL->data), ggml_nbytes(inpL));
    struct ggml_tensor * ffn_scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1);
    ffn_scale->data = malloc(ggml_nbytes(ffn_scale));
    ggml_set_f32(ffn_scale, 0.5f);

    for (int il = 0; il < n_audio_enc_layer; ++il) {
        struct ggml_tensor * cur = inpL;
        struct ggml_tensor * residual = cur;
        const audio_enc_layer layer = model.audio_enc_layers[il];
        // FFN1: layernorm
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, layer.ffn1_layer_norm_w, cur),
                    cur),
                ggml_repeat(ctx0, layer.ffn1_layer_norm_b, cur));
        // FFN1: proj
        cur = ggml_mul_mat(ctx0, layer.ffn1_w1, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, layer.ffn1_b1, cur), cur);
        cur = ggml_silu(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ffn1_w2, cur);
        cur = ggml_add(ctx0, ggml_repeat(ctx0, layer.ffn1_b2, cur), cur);
        // FFN1: * 0.5
        cur = ggml_mul(ctx0, ggml_repeat(ctx0, ffn_scale, cur), cur);
        // FFN1: + residual
        cur = ggml_add(ctx0, cur, residual);

        // self_attn: layernorm
        cur = ggml_norm(ctx0, cur, hparams.eps);
        cur = ggml_add(ctx0,
                ggml_mul(ctx0,
                    ggml_repeat(ctx0, layer.self_attn_layer_norm_w, cur),
                    cur),
                ggml_repeat(ctx0, layer.self_attn_layer_norm_b, cur));

        // self_attn: qkv
        struct ggml_tensor * Qcur = ggml_mul_mat(ctx0,
                layer.self_attn_linear_q_w,
                cur);

        Qcur = ggml_add(ctx0,
                ggml_repeat(ctx0,
                    layer.self_attn_linear_q_b,
                    Qcur),
                Qcur);

        struct ggml_tensor * Kcur = ggml_mul_mat(ctx0,
                layer.self_attn_linear_k_w,
                cur);
        Kcur = ggml_add(ctx0,
                ggml_repeat(ctx0,
                    layer.self_attn_linear_k_b,
                    Kcur),
                Kcur);

        struct ggml_tensor * Vcur = ggml_mul_mat(ctx0,
                layer.self_attn_linear_v_w,
                cur);

        Vcur = ggml_add(ctx0,
                ggml_repeat(ctx0,
                    layer.self_attn_linear_v_b,
                    Vcur),
                Vcur);
        // self_attn: rel_pos SDPA
        int32_t S = cur->ne[1];
        int32_t H = n_audio_enc_head;
        int32_t K_h = n_audio_enc_dim / H;
        
        int32_t start_index = n_ctx - S;
        int32_t end_index = n_ctx + S - 1;

        int num_indices = end_index - start_index;

        struct ggml_tensor *rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, num_indices);
        rows->data = malloc(ggml_nbytes(rows));

        for (int i = 0; i < num_indices; i++) {
            ((int32_t *)rows->data)[i] = start_index + i;
        }
        // self_attn: load pos_enc weights & compute_r
        struct ggml_tensor * r = ggml_get_rows(ctx0, model.audio_enc_pos_enc_w, rows);
        r = ggml_mul_mat(ctx0, layer.self_attn_linear_pos_w, r); // TODO: reshape
        r = ggml_dup(ctx0, ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                r,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K_h, H, S*2-1)),
                            0, 2, 1, 3));


        struct ggml_tensor * u_bias = ggml_reshape_3d(ctx0, layer.self_attn_pos_bias_u, K_h, 1, H);
        struct ggml_tensor * v_bias = ggml_reshape_3d(ctx0, layer.self_attn_pos_bias_v, K_h, 1, H);

        // (H * K_h, S) -> (K_h, H, S) -> (K_h, S, H)
        struct ggml_tensor * Q =
                    ggml_dup(ctx0, ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Qcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K_h, H, S)),
                            0, 2, 1, 3));
        struct ggml_tensor * K =
                    ggml_dup(ctx0, ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Kcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K_h, H, S)),
                            0, 2, 1, 3));
        // struct ggml_tensor * V =
        //             ggml_dup(ctx0, ggml_permute(ctx0,
        //                     ggml_cpy(ctx0,
        //                         Vcur,
        //                         ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K_h, H, S)),
        //                     1, 2, 0, 3));
        struct ggml_tensor * V =
                    ggml_dup(ctx0, ggml_permute(ctx0,
                            ggml_cpy(ctx0,
                                Vcur,
                                ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, K_h, H, S)),
                            1, 2, 0, 3));

        // (K_h, S, H)
        struct ggml_tensor * q_with_u_bias = ggml_add(ctx0, Q, u_bias);
        struct ggml_tensor * q_with_v_bias = ggml_add(ctx0, Q, v_bias);

        struct ggml_tensor * ac = ggml_mul_mat(ctx0, K, q_with_u_bias);
        struct ggml_tensor * bd = ggml_mul_mat(ctx0, r, q_with_v_bias);

        // self_attn: shift_bd
        bd = ggml_dup(ctx0, ggml_permute(ctx0, bd, 2, 1, 0, 3)); // H, S, 2S-1


        struct ggml_tensor * pad = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, H, S, 1);
        pad->data = malloc(ggml_nbytes(pad));

        pad = ggml_set_f32(pad, 0.0);
        bd = ggml_concat(ctx0, pad, bd); // bd[i][j][0] == 0
        bd = ggml_dup(ctx0, ggml_permute(ctx0, bd, 2, 1, 0, 3)); // ok -> (2S, S, H) = pytorch (H, S, 2S)

        bd = ggml_dup(ctx0, ggml_reshape_3d(ctx0, bd, S, 2*S, H));  // ok. (S, 2S, H)

        bd = ggml_remove_head_row(ctx0, bd);

        bd = ggml_reshape_3d(ctx0, bd, 2*S-1, S, H);

        bd = ggml_get_first_cols_by_rows(ctx0, bd);



        // self_attn: compute attn / weights

        struct ggml_tensor * attn_weights = ggml_add(ctx0, ac, bd);
        // inpL = ggml_sum(ctx0, attn_weights);


        struct ggml_tensor * attn_scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 1);
        attn_scale->data = malloc(ggml_nbytes(attn_scale));
        ggml_set_f32(attn_scale, 1.0 / pow(K_h, 0.5));
        attn_weights = ggml_mul(ctx0, ggml_repeat(ctx0, attn_scale, attn_weights), attn_weights);
        attn_weights = ggml_soft_max(ctx0, attn_weights);
        struct ggml_tensor * attn = ggml_mul_mat(ctx0, V, attn_weights);
        inpL = attn;
        break;


        // conv

        // ffn2

        // norm

    }

    ggml_build_forward_expand(gf, inpL);
    ggml_free(ctx0);

    return gf;
}

bool unity_eval(
        const unity_model & model,
        struct ggml_allocr * allocr,
        const int n_threads) {

    const auto & hparams = model.hparams;

    // reset the allocator to free all the memory allocated during the previous inference
    ggml_allocr_reset(allocr);

    struct ggml_cgraph * gf = unity_graph(model, allocr);

    // allocate tensors
    ggml_allocr_alloc_graph(allocr, gf);

    // run the computation
    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads);
    static std::vector<uint8_t> work_buffer;
    work_buffer.resize(plan.work_size);
    plan.work_data = work_buffer.data();
    ggml_graph_compute(gf, &plan);

    // in this case, the output tensor is the last one in the graph
    struct ggml_tensor * inpL = gf->nodes[gf->n_nodes - 1];
    for (int i = 0; i < 1000; ++i) {
        printf("%8.4f ", ((float *)(inpL->data))[i]);
    }

    return true;
}

int main(int argc, char ** argv) {
    // ggml_time_init();

    // const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        params.prompt = gpt_random_prompt(rng);
    }

    gpt_vocab vocab;
    unity_model model;

    // load the model
    {
        if (!unity_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }

    // keep this buffer alive while evaluating the model
    std::vector<uint8_t> compute_buffer;
    struct ggml_allocr * allocr = NULL;
    // allocate the compute buffer
    {
        allocr = ggml_allocr_new_measure(GGML_MEM_ALIGN);
        struct ggml_cgraph * gf = unity_graph(model, allocr);


        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf) + GGML_MEM_ALIGN;

        // recreate the allocator with the required memory
        ggml_allocr_free(allocr);
        compute_buffer.resize(mem_size);
        allocr = ggml_allocr_new(compute_buffer.data(), mem_size, GGML_MEM_ALIGN);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
    }

    if (!unity_eval(model, allocr, 1)) {
        printf("Failed to predict\n");
        return 1;
    }

    ggml_free(model.ctx);

    return 0;
}
