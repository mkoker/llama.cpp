# Scheduler Cache Call Map (2026-04-25)

Branch: rex/expert-cache-scheduler-v1
Target file: ggml/src/ggml-backend.cpp

## Primary integration block
- Scheduler split input copy loop starts around lines 1547+.
- MoE selective copy branch is around 1569+.
- Existing CUDA cache hook lookup point is around 1621+ (symbol: ggml_backend_cuda_expert_cache_copy).

## Confirmed edit anchors
- Usage gate candidate currently tied to GGML_BACKEND_BUFFER_USAGE_WEIGHTS appears around lines 916 and 1280.
- Copy path to replace/wire is in the split loop around 1547-1680.
- Current selective copy lambda exists around 1634+ (copy_experts).

## Existing cache API (ggml-cuda/expert-cache)
- Init: ggml_expert_cache_init(...)
- Free: ggml_expert_cache_free(...)
- Lookup/allocate slot: ggml_expert_cache_get(...)
- Key: (tensor_ptr, expert_idx) in ggml_expert_cache_key
- LRU internals: lru_unlink, lru_push_front, victim at lru_tail

## Integration sequence to implement
1. Derive cache key at scheduler MoE copy site.
2. Lookup cache slot for each expert slice.
3. On miss, perform H2D once, then insert/update slot metadata.
4. On hit, skip host->device copy and reuse resident slot.
5. Preserve required destination layout behavior and synchronization.
6. Add counters for hit/miss/H2D/D2D/skipped H2D.
