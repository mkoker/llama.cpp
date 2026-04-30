# Layer Hidden Export API Design (Task 2)

## Problem statement

DFlash drafting needs intermediate hidden states from selected target-model transformer layers. Current llama.cpp APIs only expose final logits/embeddings and internal KV cache, not per-layer hidden outputs. We need a stable API that can export hidden vectors for caller-selected layers during normal decode, without changing model math or introducing cross-request state leaks.

## Goals

1. Add a public C API to fetch hidden outputs for configured layer IDs after each decode/eval step.
2. Keep behavior opt-in; zero overhead when disabled.
3. Preserve compatibility with existing llama_decode() call sites.
4. Support future speculative drafters (DFlash, EAGLE-style) that consume hidden states.

## Non-goals

- No file serialization in this phase.
- No JSON/stdout formatting in core library.
- No changes to token sampling logic.

## Proposed API surface

Header additions in include/llama.h (names for Task 3 implementation):

- `void llama_set_layer_outputs(struct llama_context * ctx, const int32_t * layer_ids, size_t n_layer_ids);`
  - Registers requested transformer layer indices for export.
  - Empty list disables capture.

- `size_t llama_get_layer_outputs_count(const struct llama_context * ctx);`
  - Number of configured output layers currently captured.

- `const int32_t * llama_get_layer_outputs_ids(const struct llama_context * ctx);`
  - Pointer to internal ordered layer-id list.

- `const float * llama_get_layer_outputs(const struct llama_context * ctx, int32_t layer_id, size_t * n_tokens, size_t * n_embd);`
  - Returns pointer to contiguous F32 host buffer for the most recent decode step for `layer_id`.
  - Layout: row-major `[token_index][embd]`.
  - `n_tokens` and `n_embd` are output parameters.
  - Returns NULL if not configured or unavailable.

## Data model and memory ownership

- Capture buffers are owned by `llama_context`.
- Buffers are refreshed on each successful decode.
- Lifespan: valid until next decode call on same context, or context destruction.
- Thread safety: same as existing `llama_context` expectations (single caller per context unless externally synchronized).

## Execution path changes (high level)

1. Parse and validate selected layer IDs against model depth.
2. During graph construction/eval, for each requested layer, materialize post-layer hidden tensor to host-accessible memory.
3. Store pointers/metadata in context-side lookup table keyed by layer_id.
4. Expose retrieval through `llama_get_layer_outputs()`.

## Performance constraints

- Disabled path must not allocate capture buffers.
- Enabled path should copy only requested layers.
- Keep data in F32 public API for deterministic interop (internal compute dtype can differ).
- Reuse buffers across decode steps when shape unchanged to reduce allocator churn.

## Error handling

- Invalid layer id in setter: ignore with warning callback or return via status code in a future revision; initial implementation can clamp to strict reject + clear state.
- Retrieval before decode: return NULL.
- Retrieval for layer not requested: return NULL.

## CLI integration sketch (for Task 4)

Potential flag:
- `--layer-outputs 3,11,23,35,43`

CLI can call setter once after context init, then for each decode print compact metadata lines or hand off to downstream tooling. Core API remains transport-agnostic.

## Test strategy

- Unit/API test: configure one valid layer, run 1-token decode, verify non-NULL buffer and expected shape (`n_tokens > 0`, `n_embd == model.n_embd`).
- Multi-layer test: request 5 layers and verify all IDs retrievable.
- Negative tests: invalid layer IDs, retrieval before decode, disabled mode.
- Perf sanity: no regression when capture disabled.

## Compatibility and rollout

- Pure additive API, no behavior change for existing users.
- If symbols are gated by ABI versioning policy, add at end of public section.
- Follow-up phases wire DFlash drafter conditioning directly to these buffers.
