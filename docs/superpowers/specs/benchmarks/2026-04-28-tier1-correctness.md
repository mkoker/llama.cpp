# Tier 1 Correctness Gate — Qwen3-30B forced offload

Date: 2026-04-28 UTC
Branch: rex/expert-cache-scheduler-v1
Model: /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf
Command path: build-hip/bin/llama-completion

Gate command:
./build-hip/bin/llama-completion   -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf   -ngl 10 -ncmoe 8 --expert-cache-size 2048   -p Explain the Rayleigh-Jeans law in one paragraph.   -n 256 -no-cnv

Result: PASS
- Build target llama-cli: PASS
- Generation run completed: PASS
- Output scan for failure signatures (nan|garbage): PASS

Observed runtime metrics (from logs/expert-cache/tier1-correctness.out):
- eval throughput: 6.50 tok/s
- total tokens: 268
- expert cache final stats:
  - hits: 165844
  - misses: 158163
  - hit rate: 51.2%
  - h2d copies: 0
  - h2d bytes: 0
  - d2d copies: 175314
  - d2d bytes: 165613621248
  - skipped h2d: 46018

Notes:
- Earlier attempt failed due host disk pressure (/tmp full) and transient ROCm OOM from stale GPU allocations; resolved before final passing run.
