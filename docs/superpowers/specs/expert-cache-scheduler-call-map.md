# Expert Cache Scheduler Call Map

Status: updated after scheduler-hardening pass.

## Goal

Connect the persistent CUDA/HIP expert cache to scheduler `-ncmoe` CPU→GPU expert-copy traffic for `GGML_OP_MUL_MAT_ID`, while avoiding dense/non-MoE paths.

## Current source anchors

- `ggml/src/ggml-backend.cpp`
  - scheduler proc typedefs: `ggml_backend_sched_expert_cache_copy_fn_t`, `ggml_backend_sched_expert_cache_insert_fn_t`
  - backend proc lookup: `ggml_backend_cuda_expert_cache_copy`, `ggml_backend_cuda_expert_cache_insert`
  - MoE split copy path: `ggml_backend_sched_compute_splits()` scans split graph for `GGML_OP_MUL_MAT_ID` where `src[0] == input_cpy`
  - cache key builder: `ggml_sched_expert_cache_make_key_base()` returns false for unsupported layouts so the original selective H2D path is used instead of asserting.
  - shared ABI key definition: `ggml/src/ggml-expert-cache-key.h` is included by scheduler and CUDA/HIP cache code and has static size/offset checks.
  - cache hit path is attempted before original selective H2D copy; miss path falls back to original selective H2D and then inserts.

- `ggml/src/ggml-cuda/expert-cache.{cuh,cu}`
  - persistent arena + LRU slots
  - key fields: version, source tensor identity, backend/device route, dtype, shape/stride, expert size, expert index
  - counters: hits, misses, H2D copies/bytes, D2D copies/bytes, skipped H2D

- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - lifecycle: `ggml_backend_cuda_set_expert_cache()` reuses the arena when config is unchanged
  - scheduler-facing procs: `ggml_backend_cuda_expert_cache_copy()` and `ggml_backend_cuda_expert_cache_insert()`
  - proc export through `ggml_backend_cuda_get_proc_address()`
  - cache ownership: scheduler-side copy/insert is the single active cache owner. The former `ggml_cuda_mul_mat_id()` staging-cache path and its staging allocation proc were removed to avoid overlapping behavior and divergent counters.

## Data flow

1. Scheduler identifies split input copied from host to GPU for `MUL_MAT_ID` expert weights.
2. Scheduler pulls expert ids from `node->src[2]` and builds `used_ids`.
3. Scheduler builds a stable key base from source tensor identity/layout/backend route.
4. Scheduler calls backend cache-copy proc.
5. Backend returns true when cache handled the selected experts by D2D/H2D+D2D into `input_cpy`. It also copies the same small end padding (`min(expert_size, 512)` when the selected run is not the final expert) that the original selective H2D path copied, so kernels see equivalent trailing bytes.
6. If backend returns false, scheduler executes original selective H2D copy and then calls insert proc.

## Immediate validation risks

- `ggml_sched_expert_cache_source_tensor_id()` must be stable across graph rebuilds; source data pointer is preferred over tensor object address.
- Cache ABI structs in scheduler and CUDA header are now one shared struct with static layout checks.
- `materialized_dst` skip was removed; every cache hit materializes destination bytes each call, avoiding unproven destination lifetime assumptions.
- Build must verify HIP target before any benchmark.
