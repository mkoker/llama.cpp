#include "models.h"

llm_build_qwen3::llm_build_qwen3(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            // compute Q and K and RoPE them
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

#include "models.h"

llm_build_dflash_draft::llm_build_dflash_draft(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // DFlash takes two dense inputs:
    //  - noise embeddings for the proposed block (ubatch.embd / t_inp_embd)
    //  - concatenated target hidden states from selected target layers
    //    (5 * n_embd rows) projected by dflash.fc + dflash.hidden_norm.
    // The projected target hidden states are injected as a KV prefix in every
    // drafter attention layer.
    inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, ubatch.n_tokens);
    cb(inpL, "dflash_noise_inp", -1);
    ggml_set_input(inpL);
    res->t_inp_embd = inpL;

    ggml_tensor * target_hidden = nullptr;
    if (cross != nullptr) {
        target_hidden = build_inp_cross_embd();
    } else {
        // Smoke paths that only load/build a standalone drafter do not yet pass
        // target hidden states. Keep the graph buildable with an explicit
        // 5*n_embd dense placeholder. The real speculative path supplies
        // cross->v_embd with these rows populated from target layer captures.
        target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, model.cls->ne[0], ubatch.n_tokens);
        ggml_set_input(target_hidden);
    }
    cb(target_hidden, "dflash_target_hidden_cat", -1);

    ggml_tensor * target_ctx = build_lora_mm(model.cls, target_hidden);
    cb(target_ctx, "dflash_target_fc", -1);
    target_ctx = build_norm(target_ctx, model.cls_norm, NULL, LLM_NORM_RMS, -1);
    cb(target_ctx, "dflash_target_hidden", -1);

    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );

            ggml_tensor * Kctx = build_lora_mm(model.layers[il].wk, target_ctx, model.layers[il].wk_s);
            cb(Kctx, "Kctx", il);
            if (model.layers[il].wk_b) {
                Kctx = ggml_add(ctx0, Kctx, model.layers[il].wk_b);
                cb(Kctx, "Kctx_b", il);
            }
            Kctx = ggml_view_3d(ctx0, Kctx, n_embd_head, n_head_kv, target_ctx->ne[1],
                    ggml_row_size(Kctx->type, n_embd_head), Kctx->nb[1], 0);
            Kctx = build_norm(Kctx, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kctx, "Kctx_normed", il);

            ggml_tensor * Vctx = build_lora_mm(model.layers[il].wv, target_ctx, model.layers[il].wv_s);
            cb(Vctx, "Vctx", il);
            if (model.layers[il].wv_b) {
                Vctx = ggml_add(ctx0, Vctx, model.layers[il].wv_b);
                cb(Vctx, "Vctx_b", il);
            }
            Vctx = ggml_view_3d(ctx0, Vctx, n_embd_head, n_head_kv, target_ctx->ne[1],
                    ggml_row_size(Vctx->type, n_embd_head), Vctx->nb[1], 0);

            // KV-prefix attention: target-hidden K/V precedes proposed-token K/V.
            // The prefix uses unmasked full attention here; Task 11 validates the
            // numerical contract before the speculative loop wires cache positions.
            Kcur = ggml_concat(ctx0, Kctx, Kcur, 2);
            Vcur = ggml_concat(ctx0, Vctx, Vcur, 2);
            cb(Kcur, "kv_prefix_k", il);
            cb(Vcur, "kv_prefix_v", il);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);
            cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
            if (model.layers[il].wo_b) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
            }
        }
        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, model.layers[il].ffn_up_s,
                model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
                model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
