# Expert Cache Scheduler Call Map

Status: implementation-session note before further code edits.

## Goal

Connect the persistent CUDA/HIP expert cache to scheduler `-ncmoe` CPU→GPU expert-copy traffic for `GGML_OP_MUL_MAT_ID`, while avoiding dense/non-MoE paths.

## Current source anchors

- `ggml/src/ggml-backend.cpp`
  - scheduler proc typedefs: `ggml_backend_sched_expert_cache_copy_fn_t`, `ggml_backend_sched_expert_cache_insert_fn_t`
  - backend proc lookup: `ggml_backend_cuda_expert_cache_copy`, `ggml_backend_cuda_expert_cache_insert`
  - MoE split copy path: `ggml_backend_sched_compute_splits()` scans split graph for `GGML_OP_MUL_MAT_ID` where `src[0] == input_cpy`
  - cache key builder: `ggml_sched_expert_cache_make_key_base()`
  - cache hit path is attempted before original selective H2D copy; miss path falls back to original selective H2D and then inserts.

- `ggml/src/ggml-cuda/expert-cache.{cuh,cu}`
  - persistent arena + LRU slots
  - key fields: version, source tensor identity, backend/device route, dtype, shape/stride, expert size, expert index
  - counters: hits, misses, H2D copies/bytes, D2D copies/bytes, skipped H2D

- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - lifecycle: `ggml_backend_cuda_set_expert_cache()` reuses the arena when config is unchanged
  - scheduler-facing procs: `ggml_backend_cuda_expert_cache_copy()` and `ggml_backend_cuda_expert_cache_insert()`
  - proc export through `ggml_backend_cuda_get_proc_address()`
  - legacy/in-kernel staging path also exists inside `ggml_cuda_mul_mat_id()`; this needs care because scheduler-side and kernel-side cache paths can overlap.

## Data flow

1. Scheduler identifies split input copied from host to GPU for `MUL_MAT_ID` expert weights.
2. Scheduler pulls expert ids from `node->src[2]` and builds `used_ids`.
3. Scheduler builds a stable key base from source tensor identity/layout/backend route.
4. Scheduler calls backend cache-copy proc.
5. Backend returns true when cache handled the selected experts by D2D/H2D+D2D into `input_cpy`.
6. If backend returns false, scheduler executes original selective H2D copy and then calls insert proc.

## Immediate validation risks

- `ggml_sched_expert_cache_source_tensor_id()` must be stable across graph rebuilds; source data pointer is preferred over tensor object address.
- Cache ABI structs in scheduler and CUDA header must remain layout-compatible.
- The current `ggml_cuda_mul_mat_id()` staging cache path may duplicate scheduler-side behavior; benchmarks must prove which path is active.
- Build must verify HIP target before any benchmark.
