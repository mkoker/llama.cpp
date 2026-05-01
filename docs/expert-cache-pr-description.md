# PR draft: scheduler-driven expert cache for CPU-offloaded MoE experts

Target branch: `rex/expert-cache-scheduler-v1`
Fork branch URL: https://github.com/mkoker/llama.cpp/tree/rex/expert-cache-scheduler-v1
Base reviewed against local mission base: `origin/expert-cache`
Current HEAD before this handoff doc: `1b65786`
Commit count before this handoff doc: 23 commits
Prepared: 2026-05-01T12:00:56Z

## Summary

This branch wires the existing GPU expert cache into the ggml scheduler path used by `-ncmoe` CPU-offloaded MoE experts.

The goal is narrow: when the scheduler repeatedly copies selected MoE expert slices from host/CPU memory to GPU memory, reuse a persistent GPU-resident cache entry instead of performing redundant host-to-device copies.

Key changes:

- Extends the scheduler's weight-like split guard so `GGML_OP_MUL_MAT_ID` compute buffers can enter the MoE expert-copy path.
- Adds a scheduler-derived expert-cache key that is stable across token steps:
  - root/source tensor identity
  - backend/device route
  - dtype
  - full shape/stride layout
  - bytes per expert slice
  - expert index
- Adds persistent expert-cache lifecycle state in the CUDA/HIP backend instead of allocating per graph/token.
- Wires scheduler hit path:
  - lookup selected experts before the existing H2D copy
  - copy cache-resident GPU slots to destination on full hit
  - skip redundant H2D transfer on hit
- Wires miss path:
  - preserve the existing CPU->GPU copy behavior
  - insert copied experts into the cache
  - handle LRU eviction
- Adds proof counters for cache hits/misses, H2D copies/bytes, D2D copies/bytes, and skipped H2D copies.
- Keeps behavior inert when expert cache is disabled or no CUDA/HIP cache proc is available.

## Why this belongs in the scheduler path

The expensive path is not generic tensor copy. It is the scheduler's CPU-offloaded MoE expert slice movement created by `-ncmoe`. Placing the glue there keeps the blast radius small, avoids changing dense/non-MoE copy paths, and lets the cache key use the scheduler's tensor/backend context directly.

## Benchmark and validation results

All GPU benchmark runs used VM100's bench-lock protocol.

| Tier | Model / gate | Config | Result |
|---|---|---|---|
| Tier 1 correctness | Qwen3-30B-A3B Q4_K_M | `-ngl 10 -ncmoe 8 --expert-cache-size 2048`, 256-token generation | PASS. Coherent completion; no `nan`/`garbage` signature. 51.2% hit rate; 46,018 skipped H2D expert copies. |
| Tier 1 performance | Qwen3-30B-A3B Q4_K_M | Original gate requested `-ngl 99 -ncmoe 8 --expert-cache-size 8192` | Not accepted as perf gate. Model load hits ROCm OOM on 32GB R9700; Mike explicitly skipped this rung for now. Lower `-ngl 10` probe showed cache-on 3.82 t/s vs cache-off 3.40 t/s. |
| Tier 2 transfer | MiniMax-M2.7 UD-IQ4_XS | `-ngl 1 -ncmoe 8 -fa 1 -p 0 -n 128 -r 3`; cache off `0`, cache on `16384` | PASS. Cache off 2.49 t/s; cache on 3.03 t/s; +21.7%. |
| Tier 3 hero | Qwen3-235B-A22B Q4_K_M | `-ngl 1 -ncmoe 8 -fa 1 -p 0 -n 128 -r 1`; cache off `0`, cache on `24576` | PASS. Cache off 0.97 t/s; cache on 2.07 t/s; +113.4%. Verbose run recorded 198,197 skipped H2D expert copies. |
| Regression | Dense/non-cache path, Qwen3.6-35B-A3B Q4_K_M | no `--expert-cache-size`, `-ngl 99 -fa 1 -p 0 -n 128 -r 1` | PASS. `tg128` emitted at 71.62 t/s; validates cache-disabled path remains inert under this smoke. |

## Tier 3 headline

On Qwen3-235B-A22B Q4_K_M with forced CPU MoE offload on a 32GB R9700, expert cache improved `tg128` throughput from 0.97 tok/s to 2.07 tok/s (+113%) while recording 198,197 skipped host-to-device expert copies.

## Reproduction commands

Run from `/mnt/nvme/llama-expert-cache-rex` with:

```bash
export LD_LIBRARY_PATH=$PWD/build-hip/bin
```

### Build

```bash
cmake --build build-hip -j 16 --target llama-bench llama-cli
```

### Tier 1 correctness

```bash
./build-hip/bin/llama-completion \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 10 -ncmoe 8 --expert-cache-size 2048 \
  -p "Explain the Rayleigh-Jeans law in one paragraph." \
  -n 256 -no-cnv
```

Note: current `llama-cli` rejects `-no-cnv`; use `llama-completion` for this prompt-style correctness check.

### Tier 2 MiniMax transfer

```bash
./build-hip/bin/llama-bench \
  -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 0 \
  -p 0 -n 128 -r 3

./build-hip/bin/llama-bench \
  -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 16384 \
  -p 0 -n 128 -r 3
```

### Tier 3 Qwen3-235B hero

```bash
./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 0 \
  -p 0 -n 128 -r 1

./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 24576 \
  -p 0 -n 128 -r 1
```

### Regression smoke

```bash
./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 0 -n 128 -r 1
```

## Validation artifacts

- `docs/superpowers/specs/benchmarks/2026-04-28-tier1-correctness.md`
- `docs/superpowers/specs/benchmarks/2026-04-25-tier1-perf.md`
- `docs/superpowers/specs/benchmarks/2026-04-30-tier2-transfer.md`
- `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md`
- `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`

## Notes for Mike before opening upstream PR

- Do not open directly against `ggerganov/llama.cpp` until this branch is reviewed in the mkoker fork.
- The branch includes benchmark logs under `logs/expert-cache/` for Tier 3 evidence. If upstream maintainers prefer no logs in-tree, squash/drop those before upstream PR and keep the markdown reports only.
- The Tier 1 perf acceptance rung was skipped because the original gate's `-ngl 99` shape does not fit the 32GB R9700. The implementation still passed Tier 1 correctness, Tier 2 transfer, Tier 3 hero, and dense regression gates.

## Suggested PR title

`ggml: cache repeated CPU-offloaded MoE expert copies`

## Suggested PR body

```markdown
## Summary
- wire the CUDA/HIP expert cache into the ggml scheduler's `-ncmoe` MoE expert-copy path
- reuse persistent GPU-resident expert slices on scheduler cache hits and skip redundant H2D copies
- preserve existing CPU->GPU copy behavior on misses, then insert copied experts into an LRU cache
- add debug/proof counters for hits, misses, H2D/D2D bytes, and skipped H2D copies

## Motivation
`-ncmoe` can repeatedly transfer the same CPU-offloaded MoE expert slices to the GPU. This patch lets the scheduler reuse those expert slices from a persistent GPU cache instead of paying repeated host-to-device copy cost.

## Results
- Tier 1 correctness, Qwen3-30B forced offload: pass; 51.2% cache hit rate; 46,018 skipped H2D copies
- Tier 2 MiniMax-M2.7 forced offload: 2.49 t/s cache-off -> 3.03 t/s cache-on (+21.7%)
- Tier 3 Qwen3-235B forced offload: 0.97 t/s cache-off -> 2.07 t/s cache-on (+113.4%); 198,197 skipped H2D copies
- Dense/cache-disabled regression smoke: pass, `tg128` row emitted without cache flag

## Test plan
- `cmake --build build-hip -j 16 --target llama-bench llama-cli`
- Qwen3-30B 256-token correctness run with `-ngl 10 -ncmoe 8 --expert-cache-size 2048`
- MiniMax-M2.7 cache-off/cache-on transfer benchmark with `-ngl 1 -ncmoe 8`
- Qwen3-235B cache-off/cache-on hero benchmark with `-ngl 1 -ncmoe 8`
- Dense Qwen3.6 cache-disabled regression smoke
```
