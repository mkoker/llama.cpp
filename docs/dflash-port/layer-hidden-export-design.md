# Layer Hidden Export API Design (Task 2)

## Problem statement

DFlash drafting and related speculative decoding experiments need intermediate hidden states from selected target-model transformer layers. Current llama.cpp public APIs expose final logits, final embeddings, and KV-cache behavior, but they do not expose per-layer hidden outputs from the decode path. Downstream drafters therefore have to patch internals or re-run model fragments, which is brittle and risks changing normal llama.cpp behavior.

The design target is an opt-in API that exports selected transformer-layer hidden states during decode without changing model math, sampling, KV-cache semantics, logits behavior, embeddings behavior, or existing call-site expectations. When the feature is disabled, llama.cpp should behave as it does today.

## Goals

1. Add a public C API for configuring layer IDs whose hidden outputs should be captured.
2. Add public accessors for the configured layer list and the captured hidden-output buffers after decode.
3. Preserve `llama_decode()` compatibility and avoid changes to normal llama.cpp behavior when no layers are configured.
4. Provide host-accessible F32 buffers with explicit token and embedding dimensions for DFlash/EAGLE-style consumers.
5. Keep disabled mode allocation-free and overhead-free except for a cheap empty-configuration branch.

## Non-goals

- No file serialization, JSON formatting, stdout transport, or network transport in the core library.
- No changes to token sampling logic, logits calculation, model math, or generated token choices.
- No attempt to expose every intermediate tensor; only caller-selected transformer layer hidden states are in scope.
- No cross-context global capture state. All state belongs to the active `llama_context`.

## Proposed API surface

Header additions in `include/llama.h` for the implementation tasks:

```c
void llama_set_layer_outputs(
        struct llama_context * ctx,
        const int32_t * layer_ids,
        size_t n_layer_ids);

size_t llama_get_layer_outputs_count(
        const struct llama_context * ctx);

const int32_t * llama_get_layer_outputs_ids(
        const struct llama_context * ctx);

const float * llama_get_layer_outputs(
        const struct llama_context * ctx,
        int32_t layer_id,
        size_t * n_tokens,
        size_t * n_embd);
```

`llama_set_layer_outputs` registers the ordered list of transformer layer indices to capture. Passing `NULL` or `n_layer_ids == 0` disables capture and releases or marks capture buffers reusable for later calls. The setter validates layer IDs against the model depth, deduplicates repeated IDs, and stores the normalized list on the `llama_context`.

`llama_get_layer_outputs_count` returns the number of configured output layers. `llama_get_layer_outputs_ids` returns a pointer to the context-owned ordered layer-id list so callers can enumerate what is configured. The returned ID pointer has the same lifetime as other context-owned metadata and must not be freed or modified by the caller.

`llama_get_layer_outputs` returns a pointer to the most recent captured F32 buffer for `layer_id`, plus `n_tokens` and `n_embd` dimensions. It returns `NULL` when the layer is not configured, no successful decode has populated the layer yet, the requested layer was invalid/unavailable, or capture is disabled.

## Data model, ownership, lifetime, and layout

- `llama_context` ownership: configured layer IDs, capture metadata, and capture buffers are owned by the `llama_context`.
- Caller ownership: callers receive borrowed `const` pointers only; callers must copy the data if they need it after another decode.
- Buffer lifetime: output buffers are valid until the next decode/eval on the same context, a subsequent `llama_set_layer_outputs` call that changes capture configuration, or context destruction.
- Layout: every exported buffer uses row-major token-by-embedding layout: `data[token_index * n_embd + embedding_index]`.
- Dimensions: `n_tokens` is the token count represented by the most recent decode batch for that captured layer; `n_embd` is the model hidden width for that layer output.
- Thread safety: the feature follows existing `llama_context` thread-safety expectations. A context should have one decode/access caller at a time unless the embedding/logit/hidden-output access is externally synchronized.
- Disabled path behavior: when no layer IDs are configured, the decode path must perform no allocation, no tensor materialization for export, and no measurable overhead beyond checking an empty list.

## Execution path

1. Store capture configuration in the context: normalized layer IDs, per-layer slots, dimensions, and reusable host F32 buffers.
2. Validate layer IDs in the setter against model depth (`0 <= layer_id < n_layer`) and reject unavailable layers before any decode work is scheduled.
3. During graph construction or graph evaluation, identify the post-transformer-block hidden tensor for each requested layer. The capture point should be after the selected layer has produced the hidden state consumed by the next layer, before final norm/logit projection changes the representation.
4. Materialize the selected tensors to host-accessible F32 buffers. Internal compute may remain quantized/F16/BF16 as usual; only the public exported copy is normalized to F32.
5. On successful decode completion, publish the captured buffer pointer and dimensions in the context-side lookup table. On failed decode, leave no partially updated public buffers or mark the affected layer unavailable for that step.
6. `llama_get_layer_outputs` performs a read-only lookup by layer ID and returns the latest published pointer plus dimensions.

## Performance constraints

- No allocation/overhead when disabled: an empty configuration must bypass capture scheduling, tensor copies, and buffer setup.
- Enabled mode copies only requested layers, never all layers by default.
- Reuse buffers across decode calls when `n_tokens` and `n_embd` are unchanged; grow only when needed.
- Keep public buffers contiguous F32 for deterministic interop, even if the internal backend tensor type differs.
- Do not force extra synchronization for non-requested layers. Any backend synchronization should be scoped to requested captures only.
- Do not retain per-token history beyond the most recent decode step; long-term storage belongs to the caller.

## Error handling

- Invalid layer IDs in `llama_set_layer_outputs`: strict reject is preferred. The implementation should clear the capture configuration or ignore only invalid entries with a warning; the chosen behavior must be documented in the header comment.
- Unavailable layers at runtime: `llama_get_layer_outputs` returns `NULL`, sets `n_tokens`/`n_embd` to zero when provided, and does not expose stale data from a previous decode.
- Retrieval before decode: return `NULL` and zero dimensions.
- Retrieval for a layer not requested: return `NULL` and zero dimensions.
- Allocation or host-copy failure: decode should fail with the existing llama.cpp error-reporting path, or capture should be marked unavailable for that layer if the broader decode can safely continue. Silent stale data is not acceptable.

## CLI integration sketch (for later tasks)

A future CLI flag can map directly onto the setter:

```text
--layer-outputs 3,11,23,35,43
```

The CLI should parse the comma-separated list once after context initialization, call `llama_set_layer_outputs`, run decode normally, then either print compact metadata or hand pointers to DFlash-specific code. The core API remains transport-agnostic.

## Test strategy

- API smoke test: configure one valid layer, run a one-token decode, verify `llama_get_layer_outputs` returns non-NULL with `n_tokens > 0` and `n_embd == model hidden width`.
- Multi-layer test: request several valid layers and verify `llama_get_layer_outputs_count`, `llama_get_layer_outputs_ids`, and each layer buffer agree.
- Disabled test: call the setter with an empty list and verify decode succeeds with no capture buffers returned.
- Negative tests: invalid layer ID, duplicate layer ID, retrieval before decode, retrieval for an unrequested layer, and runtime unavailable layer behavior.
- Performance sanity: disabled capture should match baseline allocation behavior and should not materially regress decode throughput.

## Compatibility and rollout

The API is additive. Existing callers that never invoke `llama_set_layer_outputs` should see identical behavior, outputs, and performance. New symbols should be appended according to llama.cpp public ABI conventions. Follow-up implementation tasks can wire DFlash drafter conditioning to these buffers without requiring another public API redesign.
