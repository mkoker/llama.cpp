# PR draft: scheduler-driven expert cache for CPU-offloaded MoE experts

Prepared for Mike review: 2026-06-24
Clean review branch: `rex/expert-cache-scheduler-pr-clean`
Original development branch: `rex/expert-cache-scheduler-v1`
Intended upstream base: `origin/expert-cache` (`/mnt/models/llama.cpp`), commit `3d4a9a7`.
Do not open an upstream PR until Mike reviews this packet.

## Branch hygiene status

- Raw benchmark logs under `logs/expert-cache/` are intentionally not present on the clean review branch. Durable evidence is summarized in markdown under `docs/superpowers/specs/benchmarks/`.
- The original development branch had 36 commits on top of `origin/expert-cache`, including raw logs and incremental bench commits. This branch is prepared for review as a squashed clean branch.
- Recommended upstream commit structure if Mike approves:
  1. `ggml: add scheduler expert cache guard and lifecycle` — scheduler eligibility guard, stable key base, shared key ABI header, safe layout fallback, persistent cache arena lifecycle.
  2. `ggml: wire scheduler expert cache hit and miss paths` — scheduler cache-copy proc, hit materialization, miss insert/LRU path, padding preservation, removal of overlapping kernel-side staging path, D2D hit-copy coalescing.
  3. `ggml: add expert cache transfer instrumentation` — hit/miss, H2D/D2D copy/byte counters, skipped-H2D counters, debug/final summaries.
  4. `docs: document expert cache validation evidence` — call map, benchmark summaries, PR description, acceptance state.

## Current acceptance state

Accepted / ready for Mike review:

- HIP build passes on VM100 clean branch.
- Focused CTests pass on VM100 clean branch.
- Raw logs are dropped from the clean branch; benchmark evidence is retained in docs.
- Branch is based on `origin/expert-cache`, not directly on `github/master`, because the scheduler integration depends on the existing expert-cache mission base.
- Correctness hardening from the review pass is included: narrow scheduler guard, safe key fallback, shared key struct/ABI checks, one scheduler-owned cache materialization path, padding preservation, and removal of the unproven destination materialization skip.

Not accepted / needs Mike decision before upstream PR:

- Tier 1 performance target remains unmet. Latest documented cache-on result after coalesced D2D refinement is `74.07 ± 0.03 t/s` vs target `82.8 tok/s`.
- Tier 1 counters show high hit rate and skipped H2D, but remaining overhead is mandatory D2D materialization into the existing `MUL_MAT_ID` input layout plus scheduler/cache bookkeeping.
- This should be reviewed as a functional scheduler integration with strong Tier 2/Tier 3 transfer wins, not as a completed Tier 1 performance win.

## Summary

This branch wires the existing GPU expert cache into the ggml scheduler path used by `-ncmoe` CPU-offloaded MoE experts.

The narrow goal: when the scheduler repeatedly copies selected MoE expert slices from host/CPU memory to GPU memory, reuse a persistent GPU-resident cache entry instead of performing redundant host-to-device copies.

Key changes:

- Extends the scheduler copy eligibility path for host-backed `GGML_OP_MUL_MAT_ID` MoE expert tensors.
- Adds a scheduler-derived expert-cache key that is stable across token steps:
  - source/root data identity
  - backend/device route
  - dtype
  - full shape/stride layout
  - bytes per expert slice
  - expert index
- Adds persistent expert-cache lifecycle state in the CUDA/HIP backend.
- Wires scheduler hit path:
  - lookup selected experts before the existing H2D copy
  - copy cache-resident GPU slots to destination on hit
  - skip redundant H2D transfer on hit
- Wires miss path:
  - preserve the existing CPU->GPU copy behavior
  - insert copied experts into the cache
  - handle LRU eviction
- Adds proof counters for cache hits/misses, H2D copies/bytes, D2D copies/bytes, and skipped H2D copies.
- Keeps behavior inert when expert cache is disabled or no CUDA/HIP cache proc is available.

## Correctness hardening included

- Scheduler guard is now a named predicate limited to host-backed MoE expert `src[0]` tensors.
- Scheduler-side copy/insert is the only active expert-cache materialization path. The former `ggml_cuda_mul_mat_id()` staging-cache path and staging allocation proc were removed to avoid divergent keys/counters.
- Unsupported layouts return false from scheduler cache-key creation and fall back to the original selective H2D path instead of hard asserting.
- Scheduler and CUDA/HIP code share `ggml/src/ggml-expert-cache-key.h`, with static offset and size checks for `ggml_expert_cache_key_base`.
- Cache copy preserves the original selective-copy trailing padding: `min(expert_size, 512)` after each selected contiguous run unless the run reaches the final expert.
- The unproven `materialized_dst` skip optimization was removed; D2D materialization is performed every cache hit.
- Cache slots use monotonic first-fill allocation and cache-hit D2D copies coalesce contiguous destination/source runs when router-selected expert IDs are adjacent.

## Benchmark and validation summary

All GPU benchmark runs referenced here used VM100's bench-lock protocol when collected. Raw logs are intentionally not committed on the clean review branch; markdown evidence is kept under `docs/superpowers/specs/benchmarks/`.

| Tier | Model / gate | Config | Result |
|---|---|---|---|
| Tier 1 correctness | Qwen3-30B-A3B Q4_K_M | `-ngl 10 -ncmoe 8 --expert-cache-size 2048`, 256-token generation | PASS. Coherent completion; no `nan`/garbage signature. Summary: `docs/superpowers/specs/benchmarks/2026-04-28-tier1-correctness.md`. |
| Tier 1 performance | Qwen3-30B-A3B Q4_K_M | `-ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192` | NOT MET. Full GPU `93.48 ± 0.18 t/s`; cache off `30.44 ± 0.67 t/s`; cache on `72.19 ± 7.51 t/s`; coalesced-D2D refinement cache on `74.07 ± 0.03 t/s` vs target `82.8 tok/s`. Summary: `docs/superpowers/specs/benchmarks/2026-06-24-tier1-refinement.md`. |
| Tier 1 counter profile | Qwen3-30B-A3B Q4_K_M | `-ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 -p 0 -n 128 -r 1 -v` | High hit rate and skipped H2D; remaining gap is D2D materialization/bookkeeping. Summary: `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`. |
| Tier 2 transfer | MiniMax-M2.7 UD-IQ4_XS | `-ngl 1 -ncmoe 8 -fa 1 -p 0 -n 128 -r 3`; cache off `0`, cache on `16384` | PASS. Latest documented cache off `2.49 ± 0.11 t/s`; cache on `8.64 ± 0.17 t/s`; `3.47x` / about `+247%`. Summary: `docs/superpowers/specs/benchmarks/2026-04-30-tier2-transfer.md`. |
| Tier 3 hero | Qwen3-235B-A22B Q4_K_M | `-ngl 1 -ncmoe 8 -fa 1 -p 0 -n 128 -r 1`; cache off `0`, cache on `24576` | PASS. Cache off `0.97 t/s` -> cache on `2.07 t/s` (+113.4%); 198,197 skipped H2D copies. Summary: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md`. |
| Regression | Dense/non-cache path, Qwen3.6-35B-A3B Q4_K_M | no `--expert-cache-size`, `-ngl 99 -fa 1 -p 0 -n 128 -r 1` | PASS. `tg128` emitted at `71.62 t/s`; validates cache-disabled path remains inert under this smoke. Summary: `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`. |

## Reproduction commands

Run from `/mnt/nvme/llama-expert-cache-pr-clean` or `/mnt/nvme/llama-expert-cache-rex` on VM100.

### Configure/build HIP tree

```bash
cmake -S . -B build-hip-rex -DGGML_HIP=ON
cmake --build build-hip-rex -j2
```

### Focused CTests

```bash
cd build-hip-rex
ctest --output-on-failure -R '(test-backend-ops|test-arg-parser|test-llama-archs|test-chat|test-jinja|test-gguf|test-quantize-fns)'
```

### Bench protocol for any future benchmark rerun

```bash
bench-lock-acquire
trap 'bench-lock-release' EXIT
export LD_LIBRARY_PATH=$PWD/build-hip-rex/bin:${LD_LIBRARY_PATH:-}
```

Do not run benchmarks without the lock. This cleanup pass did not run new benchmarks.

### Tier 1 correctness

```bash
./build-hip-rex/bin/llama-completion \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 10 -ncmoe 8 --expert-cache-size 2048 \
  -p "Explain the Rayleigh-Jeans law in one paragraph." \
  -n 256
```

Note: current `llama-cli` rejects `-no-cnv`; use `llama-completion` for this prompt-style correctness check.

### Tier 1 performance/counter shape

```bash
./build-hip-rex/bin/llama-bench \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 \
  -p 0 -n 128 -r 3

./build-hip-rex/bin/llama-bench -v \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 \
  -p 0 -n 128 -r 1
```

### Tier 2 MiniMax transfer

```bash
./build-hip-rex/bin/llama-bench \
  -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 0 \
  -p 0 -n 128 -r 3

./build-hip-rex/bin/llama-bench \
  -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 16384 \
  -p 0 -n 128 -r 3
```

### Tier 3 Qwen3-235B hero

```bash
./build-hip-rex/bin/llama-bench \
  -m /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 0 \
  -p 0 -n 128 -r 1

./build-hip-rex/bin/llama-bench \
  -m /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 24576 \
  -p 0 -n 128 -r 1
```

## Verification performed on clean branch

Final branch: `rex/expert-cache-scheduler-pr-clean`.

- `git ls-files "logs/expert-cache/*" | wc -l` -> `0` tracked raw benchmark logs.
- `cmake -S . -B build-hip-rex -DGGML_HIP=ON` -> configured successfully with HIP/hipBLAS.
- `cmake --build build-hip-rex -j2` -> built successfully through `llama-server`.
- `cd build-hip-rex && ctest --output-on-failure -R '(test-backend-ops|test-arg-parser|test-llama-archs|test-chat|test-jinja|test-gguf|test-quantize-fns)'` -> `100% tests passed, 0 tests failed out of 11`, total real time `267.04 sec`.

## Suggested PR title

`ggml: cache repeated CPU-offloaded MoE expert copies`

## Suggested PR body

```markdown
## Summary
- wire the CUDA/HIP expert cache into the ggml scheduler's `-ncmoe` MoE expert-copy path
- reuse persistent GPU-resident expert slices on scheduler cache hits and skip redundant H2D copies
- preserve existing CPU->GPU copy behavior on misses, then insert copied experts into an LRU cache
- add debug/proof counters for hits, misses, H2D/D2D bytes, and skipped H2D copies

## Results
- Tier 1 Qwen3-30B forced-offload cache-on after coalesced-D2D refinement: `74.07 ± 0.03 t/s` vs target `82.8 tok/s` — target not met.
- Tier 2 MiniMax-M2.7 forced offload: `2.49 ± 0.11 t/s` cache-off -> `8.64 ± 0.17 t/s` cache-on (`3.47x`, about `+247%`).
- Tier 3 Qwen3-235B forced offload: `0.97 t/s` cache-off -> `2.07 t/s` cache-on (+113.4%); 198,197 skipped H2D copies.
- Dense/cache-disabled regression smoke: pass, `tg128 71.62 t/s` without cache flag.

## Test plan
- `cmake -S . -B build-hip-rex -DGGML_HIP=ON`
- `cmake --build build-hip-rex -j2`
- `cd build-hip-rex && ctest --output-on-failure -R '(test-backend-ops|test-arg-parser|test-llama-archs|test-chat|test-jinja|test-gguf|test-quantize-fns)'`
```
