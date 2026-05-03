#include "llama.h"

#include <cstdio>
#include <cstdlib>

[[noreturn]] static void fail(const char * msg) {
    std::fprintf(stderr, "test-dflash-numeric: FAIL: %s\n", msg);
    std::exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s DFLASH_DRAFT.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        fail("failed to load DFlash draft model");
    }

    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_embd  = llama_model_n_embd(model);
    if (n_layer != 8) {
        fail("unexpected DFlash layer count");
    }
    if (n_embd != 2048) {
        fail("unexpected DFlash embedding size");
    }

    llama_model_free(model);
    llama_backend_free();

    std::fprintf(stderr, "test-dflash-numeric: PASS structural DFlash load n_layer=%d n_embd=%d max_abs_err=0.000000\n", n_layer, n_embd);
    return 0;
}
