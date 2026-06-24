# Tier 2 Transfer Gate (MiniMax-M2.7 UD-IQ4_XS)

Date: 2026-04-30
Branch: rex/expert-cache-scheduler-v1

## Failure root cause
Original gate used `-ngl 99` and failed model load with ROCm OOM on R9700 32GB:
- attempted allocation: ~102760 MiB (full offload)
- result: `failed to load model`

## Stable gate parameters
Used forced offload configuration that fits VRAM and exercises CPU->GPU expert path:
- `-ngl 1 -ncmoe 8`
- cache off: `--expert-cache-size 0`
- cache on: `--expert-cache-size 16384`
- `-p 0 -n 128 -r 3`

## Results
- cache off tg128: 2.49 t/s
- cache on tg128: 3.03 t/s
- relative speedup: +21.7%

## Gate behavior
Updated gate now:
1) acquires bench lock with trap-based release
2) runs off and on variants back-to-back
3) parses tg128 t/s
4) passes only if `on > off`
