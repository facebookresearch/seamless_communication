#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "math.h"
#include "model_loader.h"
#include "fairseq2.h"

#include <thread>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sndfile.h>
#include <cstdlib>

struct unity_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    std::string model      = "seamlessM4T_medium.ggml"; // model path
    std::string tgt_lang = "eng";
    std::vector<std::string> files = {};
    bool text = false;
    SequenceGeneratorOptions opts = {
        /*beam_size*/ 5,
        /*min_seq_len*/ 1,
        /*soft_max_seq_len_a*/ 1,
        /*soft_max_seq_len_b*/ 200,
        /*hard_max_seq_len*/ 1000,
        /*len_penalty*/ 1.0,
        /*unk_penalty*/ 0.0,
        /*normalize_scores*/ true,
    };
};


void unity_print_usage(int /*argc*/, char ** argv, const unity_params & params) {
    fprintf(stderr, "usage: %s [options] file1 file2 ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  --text                text output\n");
    fprintf(stderr, "  --beam-size           beam size (default: %d)\n", params.opts.beam_size);
    fprintf(stderr, "\n");
}

std::string get_next_arg(int& i, int argc, char** argv, const std::string& flag, unity_params& params) {
    if (i + 1 < argc && argv[i + 1][0] != '-') {
        return argv[++i];
    } else {
        fprintf(stderr, "error: %s requires one argument.\n", flag.c_str());
        unity_print_usage(argc, argv, params);
        exit(0);
    }
}


bool unity_params_parse(int argc, char ** argv, unity_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            unity_print_usage(argc, argv, params);
        } else if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else if (arg == "-m" || arg == "--model") {
            params.model = get_next_arg(i, argc, argv, arg, params);
        } else if (arg == "-l" || arg == "--tgt-lang") {
            params.tgt_lang = get_next_arg(i, argc, argv, arg, params);
        } else if (arg == "--text") {
            params.text = true;
        } else if (arg == "-b" || arg == "--beam-size") {
            params.opts.beam_size = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else {
            params.files.push_back(std::string(arg));
        }
    }
    return true;
}

struct ggml_cgraph * unity_speech_encoder(
        fairseq2_model& model,
        struct ggml_tensor * speech_input) {
    ggml_context* ctx0 = model.ctx;
    ggml_cgraph* gf = ggml_new_graph(ctx0);
    ggml_tensor* seqs = StandardConformerEncoder_forward(model, "speech_encoder", speech_input, nullptr);
    seqs = ggml_dup(model.ctx, seqs);
    ggml_build_forward_expand(gf, seqs);
    return gf;
}


Hypothesis* unity_decode(
        fairseq2_model& model,
        const SequenceGeneratorOptions& opts,
        int tgt_lang_idx,
        ggml_tensor* encoder_output,
        int n_threads
) {
    SequenceGeneratorJob job = {
        opts,
        /*prefix_seq*/ nullptr,
        /*pad_idx*/model.vocab.token_to_id["<pad>"],
        /*unk_idx*/model.vocab.token_to_id["<unk>"],
        /*bos_idx*/model.vocab.token_to_id["<s>"],
        /*eos_idx*/model.vocab.token_to_id["</s>"],
        /*num_threads*/n_threads,
    };
    struct ggml_tensor * prefix_seq = ggml_new_tensor_1d(model.ctx, GGML_TYPE_I32, 2);
    ((int *)prefix_seq->data)[0]  = job.eos_idx;
    ((int *)prefix_seq->data)[1]  = tgt_lang_idx;
    job.prefix_seq = prefix_seq;
    return generate_sequence(model, job, encoder_output, nullptr, model.ctx);
}

int main(int argc, char ** argv) {

    unity_params params;

    if (unity_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fairseq2_model model;

    // load the model
    if (load_fairseq2_ggml_file(model, params.model.c_str())) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }
    int ctx_size_gb = 20;
    if (model.hparams["w2v2_encoder_config__num_encoder_layers"] == 24) {
        ctx_size_gb = 40;
    } 

    char result_str[4096];
    static std::vector<uint8_t> encoder_buf(ctx_size_gb * 1024LL * 1024LL * 1024LL);

    std::string input;
    bool interactive = params.files.size() == 0;
    auto next_file = params.files.begin();
    while (true) {
        if (interactive) {
            std::cout << "\nEnter audio_path and tgt_lang, separated by space (or 'exit' to quit):\n";
            std::getline(std::cin, input);
            if (input == "exit") {
                break;
            }
        } else {
            if (next_file == params.files.end()) break;
            input = *(next_file++);
        }
        std::istringstream iss(input);
        std::string audio_path;
        std::string tgt_lang = params.tgt_lang;
        iss >> audio_path >> tgt_lang;
        if (audio_path == "-") {
            audio_path = "/proc/self/fd/0";
        }
        std::cerr << "Translating (Transcribing) " << audio_path << " to " << tgt_lang << "\n";
        SF_INFO info;
        SNDFILE* sndfile = sf_open(audio_path.c_str(), SFM_READ, &info);
        if (!sndfile) {
            std::cerr << "Could not open file\n";
            if (interactive) continue;
            else return 1;
        }
        auto tgt_lang_ptr = model.vocab.token_to_id.find("__" + tgt_lang + "__");
        if (tgt_lang_ptr == model.vocab.token_to_id.end()) {
            std::cerr << "Unknown language " << tgt_lang << "\n";
            if (interactive) continue;
            else return 2;
        }
        int tgt_lang_idx = tgt_lang_ptr->second;

        // Load audio input
        std::vector<float> data(info.frames * info.channels); // Assume info.channels is always 1
        sf_readf_float(sndfile, data.data(), info.frames);

        // Reset the ggml_context
        model.ctx = ctx_from_buffer(encoder_buf);
        ggml_tensor* seqs = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, info.frames, 1);
        memcpy(seqs->data, data.data(), data.size() * sizeof(float));
        // Audio encoder
        ggml_cgraph* gf = unity_speech_encoder(model, seqs);
        ggml_graph_compute_with_ctx(model.ctx, gf, params.n_threads);
        ggml_tensor* encoder_output = gf->nodes[gf->n_nodes - 1];

        // Beam search decoding
        const Hypothesis* result = unity_decode(model, params.opts, tgt_lang_idx, encoder_output, params.n_threads);
    
        // Drop language and bos token.
        ggml_tensor* tokens = ggml_slice(model.ctx, result[0].seq, 0, 2, 0);

        // Collect result string
        int n = fairseq2_spm_detokenize(&model, tokens, (char*)&result_str);
        std::cout << std::string((char*)&result_str, n) << std::endl;
        ggml_free(model.ctx);
    }

    return 0;
}
