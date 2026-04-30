# Tier 2 Transfer Gate — MiniMax-M2.7 UD-IQ4_XS

Date: 2026-04-30 UTC
Branch: rex/expert-cache-scheduler-v1
Build: 260d3e8

Command pair (identical except cache size):
- cache off: `./build-hip/bin/llama-bench -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 0 -p 0 -n 128 -r 3`
- cache on:  `./build-hip/bin/llama-bench -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf -ngl 1 -ncmoe 8 -fa 1 --expert-cache-size 16384 -p 0 -n 128 -r 3`

Results:
- OFF tg128: 2.11 t/s
- ON  tg128: 3.03 t/s
- Delta: +0.92 t/s (+43.6%)

Gate condition `ON_TPS > OFF_TPS` is satisfied in this run.
