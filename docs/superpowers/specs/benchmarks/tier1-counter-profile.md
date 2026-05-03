# Tier 1 expert-cache counter profile

Generated: 2026-05-03T18:48:01+00:00
Commit: 02ef827 (564)

## Command

```bash
bench-lock-acquire
export LD_LIBRARY_PATH=$PWD/build-hip/bin
./build-hip/bin/llama-bench -v -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -fa 1 -ncmoe 8 --expert-cache-size 8192 -p 0 -n 128 -r 1 2>&1 | tee logs/expert-cache/next/tier1-counter-run-verbose.out
bench-lock-release
```

## Result

- Plain gate run: 62.57 t/s (`logs/expert-cache/next/tier1-counter-run.out`).
- Verbose counter run: 63.62 t/s (`logs/expert-cache/next/tier1-counter-run-verbose.out`).
- Cache arena: `expert cache arena initialized on device 0 — 6657 slots, 8191.2 MiB total`.
- Final counters: `ggml_expert_cache_free: expert cache stats (device 0) - hits: 22368, misses: 2400, hit rate: 90.3%, h2d copies: 2400, h2d bytes: 2380050432, d2d copies: 24768, d2d bytes: 24424022016, skipped h2d: 22368`.

## Counter-derived profile

- Cache lookups/copy decisions: 24,768 total = 22,368 hits + 2,400 misses.
- Hit rate: 90.3% (miss fraction 9.7%).
- Host-to-device traffic: 2,400 copies, 2269.8 MiB total, 0.946 MiB per expert slice.
- Device-to-device traffic: 24,768 copies, 23292.6 MiB total.
- Avoided host-to-device copies due to hits: 22,368; at the measured slice size this avoids about 20.7 GiB of H2D traffic in the tg128 run.

## Baseline context from previous locked logs

| run | tg128 t/s | log |
| --- | ---: | --- |
| full GPU (`-ncmoe 0`) | 93.48 | `logs/expert-cache/next/tier1-fullgpu.out` |
| cache off (`-ncmoe 8 --expert-cache-size 0`) | 30.44 | `logs/expert-cache/next/tier1-cacheoff.out` |
| cache on triplet (`--expert-cache-size 8192 -r 3`) | 72.19 | `logs/expert-cache/next/tier1-cacheon.out` |
| cache on counter run (`--expert-cache-size 8192 -r 1 -v`) | 63.62 | `logs/expert-cache/next/tier1-counter-run-verbose.out` |

The verbose counter run is 31.9% below the full-GPU baseline and recovers 52.6% of the cache-off to full-GPU gap.

## Cache-size sweep context

| expert-cache-size MiB | tg128 t/s | exit |
| ---: | ---: | ---: |
| 2048 | 59.03 | 0 |
| 4096 | 63.55 | 0 |
| 8192 | 63.49 | 0 |
| 12288 | 63.25 | 0 |
| 16384 | 63.61 | 0 |
| 24576 | 29.84 | 0 |

Observation: 4096-16384 MiB are effectively flat at ~63.3-63.6 t/s in single-run sweep data, so the current Tier 1 bottleneck is not simple capacity once the cache is large enough. The counters show only 2.2 GiB of real H2D traffic remains and ~20.7 GiB of H2D is skipped, while every selected expert still incurs a D2D materialization copy (~22.7 GiB total). Next optimization work should target avoidable scheduler/cache bookkeeping and D2D materialization path overhead, not larger cache sizes.
