# PR draft: scheduler-driven expert cache for CPU-offloaded MoE experts

Target branch: `rex/expert-cache-scheduler-v1`
Fork branch URL: https://github.com/mkoker/llama.cpp/tree/rex/expert-cache-scheduler-v1
Base reviewed against local mission base: `origin/expert-cache`
Current HEAD before PR-packet evidence refresh: `1b65786`
Commit count before PR-packet evidence refresh: 23 commits
Prepared: 2026-05-08T18:42Z

## Latest rerun evidence

This packet is refreshed from the latest rerun artifacts already captured in-tree. No new benchmark run was performed for this update.

- Tier 1 full GPU baseline: `logs/expert-cache/next/tier1-fullgpu.out`
- Tier 1 forced-offload cache off: `logs/expert-cache/next/tier1-cacheoff.out`
- Tier 1 forced-offload cache on triplet: `logs/expert-cache/next/tier1-cacheon.out`
- Tier 1 counter plain run: `logs/expert-cache/next/tier1-counter-run.out`
- Tier 1 counter verbose run: `logs/expert-cache/next/tier1-counter-run-verbose.out`
- Tier 1 counter profile summary: `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`
- Tier 2 latest cache off: `logs/expert-cache/next/tier2-next-off.out`
- Tier 2 latest cache on: `logs/expert-cache/next/tier2-next-on.out`
- Tier 3 hero evidence: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md` and `logs/expert-cache/tier3-*.out`
- Dense/cache-disabled regression: `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`

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

All GPU benchmark runs used VM100's bench-lock protocol. Latest rerun numbers below use the artifact paths listed in the Result column.

| Tier | Model / gate | Config | Result |
|---|---|---|---|
| Tier 1 correctness | Qwen3-30B-A3B Q4_K_M | `-ngl 10 -ncmoe 8 --expert-cache-size 2048`, 256-token generation | PASS. Coherent completion; no `nan`/`garbage` signature. 51.2% hit rate; 46,018 skipped H2D expert copies. Artifact: `docs/superpowers/specs/benchmarks/2026-04-28-tier1-correctness.md`. |
| Tier 1 performance | Qwen3-30B-A3B Q4_K_M | `-ngl 99 -fa 1`; full GPU baseline, forced-offload cache off, forced-offload cache on `--expert-cache-size 8192` | RERUN COMPLETE; target not yet met. Full GPU `93.48 ± 0.18 t/s` (`logs/expert-cache/next/tier1-fullgpu.out`); cache off `30.44 ± 0.67 t/s` (`logs/expert-cache/next/tier1-cacheoff.out`); cache on triplet `72.19 ± 7.51 t/s` (`logs/expert-cache/next/tier1-cacheon.out`) vs Tier 1 target `82.8 tok/s`, so acceptance is not yet met. Older `-ngl 10` cache-on 3.82 t/s vs cache-off 3.40 t/s is historical only and superseded by these `-ngl 99` reruns. |
| Tier 1 counter profile | Qwen3-30B-A3B Q4_K_M | `-ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 -p 0 -n 128 -r 1 -v` | Verbose counter run `63.62 t/s` (`logs/expert-cache/next/tier1-counter-run-verbose.out`; summary in `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`). Final counters: hit rate `90.3%`; hits `22,368`; misses `2,400`; skipped H2D `22,368`; real H2D `2,400` copies / `2,380,050,432` bytes; D2D `24,768` copies / `24,424,022,016` bytes. Interpretation: remaining Tier 1 bottleneck is likely scheduler/cache bookkeeping or D2D materialization, not cache capacity. |
| Tier 2 transfer | MiniMax-M2.7 UD-IQ4_XS | `-ngl 1 -ncmoe 8 -fa 1 -p 0 -n 128 -r 3`; cache off `0`, cache on `16384` | PASS. Latest rerun cache off `2.49 ± 0.11 t/s` (`logs/expert-cache/next/tier2-next-off.out`); cache on `8.64 ± 0.17 t/s` (`logs/expert-cache/next/tier2-next-on.out`); improvement `3.47x` / about `+247%`. Older `3.03 t/s` cache-on result is historical and no longer the headline. |
| Tier 3 hero | Qwen3-235B-A22B Q4_K_M | `-ngl 1 -ncmoe 8 -fa 1 -p 0 -n 128 -r 1`; cache off `0`, cache on `24576` | PASS. Cache off `0.97 t/s` -> cache on `2.07 t/s` (+113.4%). Verbose run recorded 198,197 skipped H2D expert copies. Artifacts: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md`, `logs/expert-cache/tier3-off-20260501-0005.out`, `logs/expert-cache/tier3-on-24576-20260501-0005.out`, `logs/expert-cache/tier3-on-8192-v-20260501-0010.out`. |
| Regression | Dense/non-cache path, Qwen3.6-35B-A3B Q4_K_M | no `--expert-cache-size`, `-ngl 99 -fa 1 -p 0 -n 128 -r 1` | PASS. `tg128` emitted at `71.62 t/s`; validates cache-disabled path remains inert under this smoke. Artifact: `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`. |

## Tier 1 counter interpretation

The latest Tier 1 forced-offload rerun no longer supports the earlier handoff assumption that `-ngl 99` is impossible on this host. The current state is: `-ngl 99` reruns completed, cache-on recovered most of the cache-off loss, but `72.19 ± 7.51 t/s` still misses the Tier 1 target of `82.8 tok/s`.

The counter-profile evidence (`docs/superpowers/specs/benchmarks/tier1-counter-profile.md`, backed by `logs/expert-cache/next/tier1-counter-run-verbose.out`) shows the cache is large enough to hit frequently: `90.3%` hit rate with `22,368` hits and `2,400` misses. The run skips `22,368` H2D copies and leaves only `2,400` real H2D copies / `2,380,050,432` bytes, but every selected expert still participates in D2D materialization: `24,768` D2D copies / `24,424,022,016` bytes. That points the remaining bottleneck at scheduler/cache bookkeeping or D2D materialization overhead rather than cache capacity.

## Tier 3 headline

On Qwen3-235B-A22B Q4_K_M with forced CPU MoE offload on a 32GB R9700, expert cache improved `tg128` throughput from 0.97 tok/s to 2.07 tok/s (+113%) while recording 198,197 skipped host-to-device expert copies. Artifacts: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md` and `logs/expert-cache/tier3-*.out`.

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

### Tier 1 latest performance reruns

```bash
./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 0 -n 128 -r 3
# Artifact: logs/expert-cache/next/tier1-fullgpu.out

./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 0 \
  -p 0 -n 128 -r 3
# Artifact: logs/expert-cache/next/tier1-cacheoff.out

./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 \
  -p 0 -n 128 -r 3
# Artifact: logs/expert-cache/next/tier1-cacheon.out
```

### Tier 1 counter profile

```bash
./build-hip/bin/llama-bench -v \
  -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf \
  -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 \
  -p 0 -n 128 -r 1 2>&1 | tee logs/expert-cache/next/tier1-counter-run-verbose.out
```

Profile summary: `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`.

### Tier 2 MiniMax transfer

```bash
./build-hip/bin/llama-bench \
  -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 0 \
  -p 0 -n 128 -r 3
# Artifact: logs/expert-cache/next/tier2-next-off.out

./build-hip/bin/llama-bench \
  -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 16384 \
  -p 0 -n 128 -r 3
# Artifact: logs/expert-cache/next/tier2-next-on.out
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

Artifacts: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md` and `logs/expert-cache/tier3-*.out`.

### Regression smoke

```bash
./build-hip/bin/llama-bench \
  -m /mnt/nvme/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 99 -fa 1 -p 0 -n 128 -r 1
```

Artifact: `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`.

## Validation artifacts

- `logs/expert-cache/next/tier1-fullgpu.out`
- `logs/expert-cache/next/tier1-cacheoff.out`
- `logs/expert-cache/next/tier1-cacheon.out`
- `logs/expert-cache/next/tier1-counter-run.out`
- `logs/expert-cache/next/tier1-counter-run-verbose.out`
- `logs/expert-cache/next/tier2-next-off.out`
- `logs/expert-cache/next/tier2-next-on.out`
- `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`
- `docs/superpowers/specs/benchmarks/2026-04-28-tier1-correctness.md`
- `docs/superpowers/specs/benchmarks/2026-04-30-tier2-transfer.md` (historical; superseded for headline by `logs/expert-cache/next/tier2-next-*.out`)
- `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md`
- `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`

## Notes for Mike before opening upstream PR

- Do not open directly against `ggerganov/llama.cpp` until this branch is reviewed in the mkoker fork.
- The branch includes benchmark logs under `logs/expert-cache/` for Tier 3 evidence. If upstream maintainers prefer no logs in-tree, squash/drop those before upstream PR and keep the markdown reports only.
- The original Tier 1 perf handoff said `-ngl 99` was skipped because of 32GB R9700 OOM. That is historical only. Later reruns completed at `-ngl 99`: full GPU `93.48 ± 0.18 t/s`, forced-offload cache off `30.44 ± 0.67 t/s`, forced-offload cache on `72.19 ± 7.51 t/s`; Tier 1 target `82.8 tok/s` is not yet met.

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
- Tier 1 Qwen3-30B forced offload, latest `-ngl 99` rerun: full GPU `93.48 ± 0.18 t/s`; cache off `30.44 ± 0.67 t/s`; cache on `72.19 ± 7.51 t/s` vs Tier 1 target `82.8 tok/s`, so the target is not yet met. Artifacts: `logs/expert-cache/next/tier1-fullgpu.out`, `logs/expert-cache/next/tier1-cacheoff.out`, `logs/expert-cache/next/tier1-cacheon.out`.
- Tier 1 counters: verbose run `63.62 t/s`; `90.3%` hit rate; hits `22,368`; misses `2,400`; skipped H2D `22,368`; real H2D `2,400` copies / `2,380,050,432` bytes; D2D `24,768` copies / `24,424,022,016` bytes. Remaining overhead appears to be scheduler/cache bookkeeping or D2D materialization, not cache capacity. Artifacts: `logs/expert-cache/next/tier1-counter-run-verbose.out`, `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`.
- Tier 2 MiniMax-M2.7 forced offload, latest rerun: `2.49 ± 0.11 t/s` cache-off -> `8.64 ± 0.17 t/s` cache-on (`3.47x`, about `+247%`). Artifacts: `logs/expert-cache/next/tier2-next-off.out`, `logs/expert-cache/next/tier2-next-on.out`.
- Tier 3 Qwen3-235B forced offload: `0.97 t/s` cache-off -> `2.07 t/s` cache-on (+113.4%); 198,197 skipped H2D copies. Artifact: `docs/superpowers/specs/benchmarks/2026-04-25-tier3-hero.md`.
- Dense/cache-disabled regression smoke: pass, `tg128 71.62 t/s` without cache flag. Artifact: `docs/superpowers/specs/benchmarks/2026-04-25-regressions.md`.

## Test plan
- `cmake --build build-hip -j 16 --target llama-bench llama-cli`
- Qwen3-30B Tier 1 latest rerun evidence from `logs/expert-cache/next/tier1-fullgpu.out`, `logs/expert-cache/next/tier1-cacheoff.out`, and `logs/expert-cache/next/tier1-cacheon.out`: target not yet met (`72.19` vs `82.8 tok/s`).
- Qwen3-30B Tier 1 counter profile from `logs/expert-cache/next/tier1-counter-run-verbose.out` and `docs/superpowers/specs/benchmarks/tier1-counter-profile.md`: high hit rate, but D2D/bookkeeping overhead remains.
- MiniMax-M2.7 Tier 2 cache-off/cache-on transfer benchmark with `-ngl 1 -ncmoe 8`: latest rerun `2.49 -> 8.64 t/s`.
- Qwen3-235B Tier 3 cache-off/cache-on hero benchmark with `-ngl 1 -ncmoe 8`: `0.97 -> 2.07 t/s`.
- Dense Qwen3.6 cache-disabled regression smoke: `tg128 71.62 t/s`.
```
