# Tier 1 performance refinement: expert-cache scheduler integration

Date: 2026-06-24
Repo: `/mnt/nvme/llama-expert-cache-rex`
Branch: `rex/expert-cache-scheduler-v1`
Guard rails: production `llama-server` untouched; `switch-model` not run; GPU benchmarks run under `bench-lock`.

## Starting evidence

Existing locked artifacts showed:

| run | result | artifact |
| --- | ---: | --- |
| full GPU | 93.48 ± 0.18 t/s | `logs/expert-cache/next/tier1-fullgpu.out` |
| forced-offload cache off | 30.44 ± 0.67 t/s | `logs/expert-cache/next/tier1-cacheoff.out` |
| forced-offload cache on | 72.19 ± 7.51 t/s | `logs/expert-cache/next/tier1-cacheon.out` |
| verbose counter | 63.62 t/s | `logs/expert-cache/next/tier1-counter-run-verbose.out` |

Counter profile showed high cache utility but heavy device materialization: 90.3% hit rate, 22,368 skipped H2D expert copies, only 2.38 GB real H2D traffic, but 24.42 GB D2D traffic. Cache-size sweep from 4096-16384 MiB was flat, so capacity was not the limiter.

## Profiling conclusion

- Duplicate cache path behavior: ruled out by current code. `ggml_cuda_mul_mat_id()` no longer owns a second staging cache; scheduler-side copy/insert is the only active cache path.
- Cache capacity: ruled out by sweep flatness and high hit rates.
- Cache lookup locking/bookkeeping: present, but prior hardening already avoids LRU mutation for large caches on hits.
- D2D materialization: still mandatory in the current design. MUL_MAT_ID consumes a normal materialized `src0` tensor, so every selected expert must be copied from a cache slot into the destination tensor even on a cache hit.

## Minimal optimization attempted

Implemented a narrow D2D launch-count optimization in `ggml/src/ggml-cuda/expert-cache.*`:

1. Added monotonic first-fill slot allocation (`next_free_slot`) so full-tensor warm fills store expert IDs in contiguous cache-slot order instead of reverse LRU-tail order.
2. Coalesced cache-hit D2D materialization for adjacent selected expert IDs whose cache slots are also adjacent.

This preserves the scheduler contract and correctness: it only changes how cache-hit D2D copies are batched when source and destination ranges are both contiguous. Sparse selections still use separate D2D copies.

## Verification

Build:

```text
cmake --build build-hip-rex -j2 --target llama-bench llama-cli
# passed; built llama-bench and llama-cli; build id a692aff-dirty during local test
```

Focused tests:

```text
cd build-hip-rex && ctest --output-on-failure -R '(test-backend-ops|test-arg-parser|test-llama-archs|test-chat|test-jinja|test-gguf|test-quantize-fns)'
# 100% tests passed, 0 tests failed out of 11, total real time 264.85 sec
```

Locked benchmark commands:

```bash
bench-lock-acquire
trap bench-lock-release EXIT
export LD_LIBRARY_PATH=$PWD/build-hip-rex/bin:${LD_LIBRARY_PATH:-}
./build-hip-rex/bin/llama-bench -v   -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf   -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192   -p 0 -n 128 -r 1 2>&1 | tee logs/expert-cache/next/tier1-coalesced-counter-run-verbose.out

./build-hip-rex/bin/llama-bench   -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf   -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192   -p 0 -n 128 -r 3 2>&1 | tee logs/expert-cache/next/tier1-coalesced-cacheon.out
```

Results:

| run | result | artifact |
| --- | ---: | --- |
| coalesced verbose counter | 72.97 t/s | `logs/expert-cache/next/tier1-coalesced-counter-run-verbose.out` |
| coalesced cache-on triplet | 74.07 ± 0.03 t/s | `logs/expert-cache/next/tier1-coalesced-cacheon.out` |

Final coalesced counters:

```text
ggml_expert_cache_free: expert cache stats (device 0) - hits: 24768, misses: 3096, hit rate: 88.9%, h2d copies: 3072, h2d bytes: 3029336064, d2d copies: 23442, d2d bytes: 24424022016, skipped h2d: 24744
```

## Postmortem / acceptance disposition

The refinement improves the latest Tier 1 cache-on triplet from `72.19 ± 7.51` to `74.07 ± 0.03`, but the Tier 1 target remains unmet (`82.8 tok/s`).

The proof point is the D2D counter behavior: bytes are unchanged at 24.42 GB and copy calls fall only 5.4% (`24,768 -> 23,442`) after coalescing. That means Qwen3-30B router selections are mostly sparse, so scheduler-side coalescing cannot remove most materialization overhead.

Target-level performance likely requires a larger design change, not another cache-size tweak:

- teach MUL_MAT_ID kernels to consume expert cache slots directly, bypassing the materialized `src0` tensor; or
- add a backend gather/materialization kernel that batches sparse expert slot copies more efficiently than thousands of scheduler-issued `cudaMemcpyAsync` D2D operations.

Tier 2/Tier 3 positive transfer evidence and the dense cache-disabled smoke remain preserved in existing artifacts:

- Tier 2: `logs/expert-cache/next/tier2-next-off.out`, `logs/expert-cache/next/tier2-next-on.out`
- Tier 3: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md`, `logs/expert-cache/tier3-*.out`
- Dense/cache-disabled: `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`
