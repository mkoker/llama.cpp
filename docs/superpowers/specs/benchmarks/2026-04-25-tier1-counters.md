# Tier-1 Cache Counter Validation (2026-04-25)

Branch: rex/expert-cache-scheduler-v1
Build: d743d2f (worktree includes uncommitted counter instrumentation)
Model: /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf
Command: llama-bench -v -fa 1 -ngl 999 -ncmoe 8 -p 64 -n 16 -r 1

Bench-lock protocol used for all runs.

## Throughput
- Cache OFF (--expert-cache-size 0): tg16 = 30.51 tok/s
- Cache ON  (--expert-cache-size 4096): tg16 = 44.27 tok/s

## Cache ON counter evidence (from ggml_expert_cache_free logs)
Run A:
- hits: 1773
- misses: 2427
- hit rate: 42.2%
- h2d copies: 2427
- h2d bytes: 2,404,749,312
- d2d copies: 4200
- d2d bytes: 4,166,406,144
- skipped h2d due to hit: 1773

Run B:
- hits: 1833
- misses: 1431
- hit rate: 56.2%
- h2d copies: 1431
- h2d bytes: 1,415,688,192
- d2d copies: 3264
- d2d bytes: 3,218,669,568
- skipped h2d due to hit: 1833

Interpretation:
- Cache hits are directly reducing host->device expert copies (skipped_h2d_due_to_hit tracks this).
- D2D copies replace repeated H2D transfers for reused experts.
