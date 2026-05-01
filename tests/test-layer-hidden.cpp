#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

[[noreturn]] static void fail(const char * msg) {
    fprintf(stderr, "test-layer-hidden: FAIL: %s\n", msg);
    std::exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    const char * model_path = argv[1];

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;

    llama_model_ptr model(llama_model_load_from_file(model_path, mparams));
    if (!model) {
        fail("failed to load model");
    }

    const int32_t n_layer = llama_model_n_layer(model.get());
    if (n_layer <= 4) {
        fail("model has too few layers for layer-hidden API test");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 16;
    cparams.n_ubatch = 16;
    cparams.no_perf = true;

    llama_context_ptr ctx(llama_init_from_model(model.get(), cparams));
    if (!ctx) {
        fail("failed to create context");
    }

    const int32_t layer_a = std::min<int32_t>(3, n_layer - 1);
    const int32_t layer_b = std::min<int32_t>(11, n_layer - 1);
    if (layer_a == layer_b) {
        fail("test selected duplicate valid layers");
    }

    const int32_t requested[] = { layer_a, layer_b, layer_a, -1, n_layer + 1000 };
    llama_set_layer_outputs(ctx.get(), requested, sizeof(requested) / sizeof(requested[0]));

    const size_t configured = llama_get_layer_outputs_count(ctx.get());
    if (configured != 2) {
        fprintf(stderr, "configured=%zu expected=2\n", configured);
        fail("layer output configuration did not deduplicate/filter ids");
    }

    const int32_t * ids = llama_get_layer_outputs_ids(ctx.get());
    if (ids == nullptr || ids[0] != layer_a || ids[1] != layer_b) {
        fprintf(stderr, "ids=%p first=%d second=%d expected=%d,%d\n",
                (const void *) ids, ids ? ids[0] : -999, ids ? ids[1] : -999, layer_a, layer_b);
        fail("configured layer output ids are wrong");
    }

    size_t pre_tokens = 123;
    size_t pre_embd = 456;
    if (llama_get_layer_outputs(ctx.get(), layer_a, &pre_tokens, &pre_embd) != nullptr || pre_tokens != 0 || pre_embd != 0) {
        fail("layer output unexpectedly available before decode");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const std::string prompt = "Test hidden states.";
    std::vector<llama_token> tokens(32);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize((size_t) -n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), true, true);
    }
    if (n_tokens <= 0) {
        fail("failed to tokenize prompt");
    }
    tokens.resize((size_t) n_tokens);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, true);
    }

    const int rc = llama_decode(ctx.get(), batch);
    llama_batch_free(batch);
    if (rc != 0) {
        fail("llama_decode failed");
    }

    std::vector<float> first_layer_data;

    for (const int32_t layer_id : { layer_a, layer_b }) {
        size_t got_tokens = 0;
        size_t got_embd = 0;
        const float * data = llama_get_layer_outputs(ctx.get(), layer_id, &got_tokens, &got_embd);
        if (data == nullptr) {
            fprintf(stderr, "layer=%d\n", layer_id);
            fail("missing captured layer output");
        }
        if (got_tokens != (size_t) n_tokens) {
            fprintf(stderr, "layer=%d got_tokens=%zu expected=%d\n", layer_id, got_tokens, n_tokens);
            fail("captured layer token count mismatch");
        }
        if (got_embd != (size_t) llama_model_n_embd(model.get())) {
            fprintf(stderr, "layer=%d got_embd=%zu expected=%d\n", layer_id, got_embd, llama_model_n_embd(model.get()));
            fail("captured layer embedding size mismatch");
        }

        const size_t n = got_tokens * got_embd;
        double sum_abs = 0.0;
        for (size_t i = 0; i < std::min<size_t>(n, 4096); ++i) {
            if (!std::isfinite(data[i])) {
                fprintf(stderr, "layer=%d index=%zu value=%f\n", layer_id, i, data[i]);
                fail("captured layer output contains non-finite value");
            }
            sum_abs += std::fabs((double) data[i]);
        }
        if (sum_abs == 0.0) {
            fprintf(stderr, "layer=%d checked=%zu\n", layer_id, std::min<size_t>(n, 4096));
            fail("captured layer output is all zero in checked prefix");
        }

        fprintf(stderr, "layer_%d tokens=%zu embd=%zu sum_abs_prefix=%.6f\n", layer_id, got_tokens, got_embd, sum_abs);

        if (layer_id == layer_a) {
            first_layer_data.assign(data, data + n);
        } else {
            if (first_layer_data.size() != n) {
                fail("captured layer output sizes differ unexpectedly");
            }
            double max_abs_diff = 0.0;
            for (size_t i = 0; i < n; ++i) {
                max_abs_diff = std::max(max_abs_diff, std::fabs((double) first_layer_data[i] - (double) data[i]));
            }
            fprintf(stderr, "layer_%d_vs_%d max_abs_diff=%.9f\n", layer_a, layer_b, max_abs_diff);
            if (max_abs_diff == 0.0) {
                fail("captured outputs for two different layers are identical");
            }
        }
    }

    llama_set_layer_outputs(ctx.get(), nullptr, 0);
    if (llama_get_layer_outputs_count(ctx.get()) != 0 || llama_get_layer_outputs_ids(ctx.get()) != nullptr) {
        fail("failed to clear layer output configuration");
    }

    ctx.reset();
    model.reset();
    llama_backend_free();

    fprintf(stderr, "test-layer-hidden: PASS\n");
    return 0;
}
