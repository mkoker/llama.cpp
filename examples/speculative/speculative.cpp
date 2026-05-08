#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.draft.mparams.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // load the draft model
    params.devices = params.speculative.draft.devices;
    params.model = params.speculative.draft.mparams;
    params.n_gpu_layers = params.speculative.draft.n_gpu_layers;
    if (params.speculative.draft.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.draft.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.draft.tensor_buft_overrides;

    auto llama_init_dft = common_init_from_params(params);

    model_dft = llama_init_dft->model();
    ctx_dft   = llama_init_dft->context();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    const bool dflash_no_vocab_dft = llama_vocab_n_tokens(vocab_dft) == 0;

    if (!dflash_no_vocab_dft && vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (!dflash_no_vocab_dft && (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    )) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    if (!dflash_no_vocab_dft) {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // DFlash/block speculative decoding proposes a fixed-size block per drafter pass.
    int n_draft = params.speculative.draft.n_max;
    if (n_draft <= 0) {
        LOG_ERR("%s: --spec-draft-n-max/--draft must be > 0\n", __func__);
        return 1;
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);


    if (dflash_no_vocab_dft) {
        LOG_INF("DFlash no-vocab draft model loaded; running target decode smoke path\n");

        const auto t_dec_start_no_vocab = ggml_time_us();
        while ((params.n_predict < 0 || n_predict < params.n_predict) && !has_eos) {
            llama_token token_id = common_sampler_sample(smpl, ctx_tgt, 0);
            common_sampler_accept(smpl, token_id, true);

            LOG("%s", common_token_to_piece(ctx_tgt, token_id).c_str());
            ++n_predict;

            if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                has_eos = true;
                break;
            }

            llama_decode(ctx_tgt, llama_batch_get_one(&token_id, 1));
        }
        const auto t_dec_end_no_vocab = ggml_time_us();

        LOG("\n\n");
        LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
        LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end_no_vocab - t_dec_start_no_vocab) / 1e6f, n_predict  / ((t_dec_end_no_vocab - t_dec_start_no_vocab) / 1e6f));
        LOG_INF("\n");
        LOG_INF("n_draft   = %d\n", n_draft);
        LOG_INF("n_predict = %d\n", n_predict);
        LOG_INF("n_decoded = %d\n", n_predict);
        LOG_INF("n_drafted = %d\n", 0);
        LOG_INF("n_accept  = %d\n", 0);
        LOG_INF("accept    = %.3f%%\n", 0.0);

        LOG_INF("\n");
        LOG_INF("draft:\n\n");
        llama_perf_context_print(ctx_dft);

        LOG_INF("\n");
        LOG_INF("target:\n\n");
        common_perf_print(ctx_tgt, smpl);

        common_sampler_free(smpl);
        llama_backend_free();
        LOG("\n\n");
        return 0;
    }

    if (n_seq_dft > 1) {
        LOG_ERR("%s: block speculative path currently supports --parallel 1 only; got --parallel %d\n", __func__, n_seq_dft);
        common_sampler_free(smpl);
        llama_backend_free();
        return 1;
    }

    // single-sequence block speculative path: draft a whole block, target-decode it once, then verify sequentially
    struct common_sampler * smpl_dft = common_sampler_clone(smpl);

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    const auto t_dec_start = ggml_time_us();

    while ((params.n_predict < 0 || n_predict < params.n_predict) && !has_eos) {
        // First generate one lossless target token. This seed token is included in the target block
        // decode so the first drafted token is verified from decoded target logits at batch index 0.
        llama_token token_seed = common_sampler_sample(smpl, ctx_tgt, 0);
        common_sampler_accept(smpl, token_seed, true);

        std::string token_seed_str = common_token_to_piece(ctx_tgt, token_seed);
        LOG("%s", token_seed_str.c_str());
        ++n_predict;

        if (llama_vocab_is_eog(vocab_tgt, token_seed)) {
            has_eos = true;
            break;
        }

        if (params.n_predict >= 0 && n_predict >= params.n_predict) {
            break;
        }

        // Keep the draft context/sampler synchronized with the target seed before proposing a block.
        common_batch_clear(batch_dft);
        common_batch_add(batch_dft, token_seed, n_past_dft, { 0 }, true);
        llama_decode(ctx_dft, batch_dft);
        ++n_past_dft;

        common_sampler_free(smpl_dft);
        smpl_dft = common_sampler_clone(smpl);

        int n_block = n_draft;
        if (params.n_predict >= 0) {
            n_block = std::min(n_block, params.n_predict - n_predict);
        }
        n_block = std::min(n_block, (int) llama_n_batch(ctx_tgt) - 1);
        if (n_block <= 0) {
            break;
        }

        std::vector<llama_token> proposed;
        proposed.reserve(n_block);

        // Block draft phase: advance the draft model token-by-token, but do not target-decode yet.
        for (int i = 0; i < n_block; ++i) {
            llama_token token_dft = common_sampler_sample(smpl_dft, ctx_dft, 0);
            common_sampler_accept(smpl_dft, token_dft, true);

            proposed.push_back(token_dft);
            ++n_drafted;

            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_dft, n_past_dft, { 0 }, true);
            llama_decode(ctx_dft, batch_dft);
            ++n_past_dft;
        }

        // Target-decode the seed plus the whole proposed block once. Logits at batch index i verify proposed[i].
        common_batch_clear(batch_tgt);
        common_batch_add(batch_tgt, token_seed, n_past_tgt, { 0 }, true);
        for (int i = 0; i < (int) proposed.size(); ++i) {
            common_batch_add(batch_tgt, proposed[i], n_past_tgt + 1 + i, { 0 }, true);
        }
        llama_decode(ctx_tgt, batch_tgt);

        ++n_past_tgt; // seed token is now committed in the target KV cache
        const int pos_block_start = n_past_tgt;

        bool mismatch = false;
        for (int i = 0; i < (int) proposed.size(); ++i) {
            llama_token token_tgt = common_sampler_sample(smpl, ctx_tgt, i);
            common_sampler_accept(smpl, token_tgt, true);

            const std::string token_tgt_str = common_token_to_piece(ctx_tgt, token_tgt);

            if (params.sampling.temp <= 0 && token_tgt == proposed[i]) {
                LOG_DBG("block draft token %d (%d, '%s') accepted\n", i, token_tgt, token_tgt_str.c_str());
                LOG("%s", token_tgt_str.c_str());
                ++n_accept;
                ++n_predict;
                ++n_past_tgt;

                if (llama_vocab_is_eog(vocab_tgt, token_tgt)) {
                    has_eos = true;
                    break;
                }

                if (params.n_predict >= 0 && n_predict >= params.n_predict) {
                    break;
                }

                continue;
            }

            // Greedy lossless verification accepts only exact matches. For non-greedy sampling this
            // conservative block path also falls back on the sampled target token at the first mismatch.
            LOG_DBG("block draft token %d rejected: draft %d '%s', target %d '%s'\n",
                    i, proposed[i], common_token_to_piece(ctx_tgt, proposed[i]).c_str(), token_tgt, token_tgt_str.c_str());
            LOG("%s", token_tgt_str.c_str());
            ++n_predict;

            const int pos_mismatch = pos_block_start + i;
            llama_memory_seq_rm(mem_tgt, 0, pos_mismatch, -1);
            common_batch_clear(batch_tgt);
            common_batch_add(batch_tgt, token_tgt, pos_mismatch, { 0 }, true);
            llama_decode(ctx_tgt, batch_tgt);

            llama_memory_seq_rm(mem_dft, 0, pos_mismatch, -1);
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, token_tgt, pos_mismatch, { 0 }, true);
            llama_decode(ctx_dft, batch_dft);

            n_past_tgt = pos_mismatch + 1;
            n_past_dft = pos_mismatch + 1;

            common_sampler_free(smpl_dft);
            smpl_dft = common_sampler_clone(smpl);

            if (llama_vocab_is_eog(vocab_tgt, token_tgt)) {
                has_eos = true;
            }
            mismatch = true;
            break;
        }

        if (!mismatch) {
            n_past_dft = n_past_tgt;
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_decoded = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_sampler_free(smpl_dft);

    llama_batch_free(batch_tgt);
    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
