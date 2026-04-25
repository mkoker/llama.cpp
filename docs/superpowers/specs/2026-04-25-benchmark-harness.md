# Expert Cache Benchmark Harness (2026-04-25)

Branch: rex/expert-cache-scheduler-v1
Binary: /mnt/nvme/llama.cpp/build-hip/bin/llama-bench
Rule: always set LD_LIBRARY_PATH to binary dir and pass -fa 1.
Lock protocol: bench-lock-acquire before benchmark runs, bench-lock-release after.

## Environment template
export BIN=/mnt/nvme/llama.cpp/build-hip/bin/llama-bench
export LD_LIBRARY_PATH=/mnt/nvme/llama.cpp/build-hip/bin:

## Tier 1 (DEV) — Qwen3-30B-A3B
Model: /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf
Target: compare full GPU baseline vs forced offload cache-off/cache-on.

Full GPU baseline:
 -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf -fa 1 -ngl 999 -ncmoe 0 -p 512 -n 128 -r 3

Forced offload (cache off):
 -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf -fa 1 -ngl 999 -ncmoe 8 -p 512 -n 128 -r 3 --expert-cache-size 0

Forced offload (cache on):
 -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf -fa 1 -ngl 999 -ncmoe 8 -p 512 -n 128 -r 3 --expert-cache-size 4096

## Tier 2 (VALIDATE) — MiniMax-M2.7 UD-IQ4_XS
Model: /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf

Cache off:
 -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf -fa 1 -ngl 999 -p 512 -n 128 -r 3 --expert-cache-size 0

Cache on:
 -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf -fa 1 -ngl 999 -p 512 -n 128 -r 3 --expert-cache-size 4096

## Tier 3 (HERO) — Qwen3-235B-A22B
Model: /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf

Cache off:
 -m /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf -fa 1 -ngl 999 -p 512 -n 128 -r 3 --expert-cache-size 0

Cache on:
 -m /mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf -fa 1 -ngl 999 -p 512 -n 128 -r 3 --expert-cache-size 4096

## Notes
- Keep prompts/tokens/repetitions fixed between cache-off/cache-on comparisons.
- Collect benchmark logs in docs/superpowers/specs/benchmarks/.
- If lock cannot be acquired, defer benchmark and retry later.
