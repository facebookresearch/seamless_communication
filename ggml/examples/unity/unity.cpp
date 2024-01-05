#include "ggml/ggml.h"
#include "ggml/ggml-alloc.h"

#include "math.h"
#include "model_loader.h"
#include "fairseq2.h"
#include "lib/unity_lib.h"
#include <sndfile.h>
#include <cstdlib>
#include "ggml-alloc.h"
#include <numeric>
#include <algorithm>

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
        /*mem_mb*/ 512,
    };
    bool verbose = false;
};


void unity_print_usage(int /*argc*/, char ** argv, const unity_params & params) {
    fprintf(stderr, "usage: %s [options] file1 file2 ...\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -v, --verbose         Print out word level confidence score and LID score (default: off)");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model.c_str());
    fprintf(stderr, "  --text                text output\n");
    fprintf(stderr, "  --beam-size           beam size (default: %d)\n", params.opts.beam_size);
    fprintf(stderr, "  -M, --mem             memory buffer, increase for long inputs (default: %d)\n", params.opts.mem_mb);
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
        } else if (arg == "-v" || arg == "--verbose") {
            params.verbose = true;
        } else if (arg == "-M" || arg == "--mem") {
            params.opts.mem_mb = std::stoi(get_next_arg(i, argc, argv, arg, params));
        } else {
            params.files.push_back(std::string(arg));
        }
    }
    return true;
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

    // The ctx_size_mb mostly depends of input length and model dim.
    int ctx_size_mb = params.opts.mem_mb;
    auto encoder_buf = std::vector<uint8_t>(8 * 1024 * 1024); // Only tensor metadata goes in there
    auto encoder_fwd_buf = std::vector<uint8_t>(ctx_size_mb * 1024 * 1024 / 2);
    ggml_allocr* fwd_alloc = ggml_allocr_new(encoder_fwd_buf.data(), encoder_fwd_buf.capacity(), 8);
    char result_str[4096];

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
        // Load audio input
        GGML_ASSERT(info.samplerate == 16000);
        GGML_ASSERT(info.channels == 1);
        // stop at 30s. Ideally we should chunk input audio, but this will prevent most obvious OOM.
        int n_frames = std::min(info.samplerate * 30, (int)info.frames);
        std::vector<float> data(n_frames * info.channels);
        sf_readf_float(sndfile, data.data(), n_frames);

        Result result = unity_eval(model, data, params.opts, tgt_lang, params.n_threads, ctx_size_mb);
        std::string concat_transcription = std::accumulate(std::next(result.transcription.begin()), result.transcription.end(), result.transcription[0],
            [](const std::string& a, const std::string& b) {
                return a + " " + b;
            }
        );
        if (params.verbose) {
            std::cout << "Final transcription: " << concat_transcription << std::endl;
            std::cout << std::endl;
            std::cout << "Word level confidence score:" << std::endl;
            for (size_t i = 0; i < result.transcription.size(); ++i) {
                std::cout << "Word: " << result.transcription[i] << " | Score: " << result.word_confidence_scores[i] << std::endl;
            }
            std::cout << std::endl;
            std::cout << "LID scores: " << std::endl;
            for (const auto& kv : result.lid_scores) {
                std::cout << "Language: " << kv.first << "| Score: " << kv.second << std::endl;
            }
        } else {
            std::cout << concat_transcription << std::endl;
        }
    }

    return 0;
}
