# Expert Cache Benchmark Harness (Tiered)

Purpose
- Provide reproducible, lock-safe benchmark commands for expert-cache evaluation across Tier 1/2/3.
- Keep run parameters deterministic and log all command outputs for comparability.

Rules
- Always acquire benchmark lock before GPU benchmark runs on VM 100.
- Always release lock in a trap, even on errors.
- Use fixed prompt/gen token lengths and fixed repetitions.
- Compare cache-off vs cache-on at identical settings.

Script
- Path: `scripts/expert-cache/run-tier-bench.sh`
- Modes:
  - `--tier tier1|tier2|tier3`
  - `--cache on|off`
  - `--reps N`
  - `--prompt-tokens N`
  - `--gen-tokens N`
  - `--dry-run`

Tier presets
- tier1: `/mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf`
  - baseline mode: `-ncmoe 0` for full-GPU reference
  - offload mode: `-ncmoe 8` for scheduler path stress
- tier2: `/mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf`
- tier3: `/mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf`

Output
- Logs: `logs/expert-cache/<timestamp>-<tier>-cache-<on|off>.log`
- Header records exact command and all run parameters.

Usage examples
- Tier1 cache-off (forced-offload):
  `scripts/expert-cache/run-tier-bench.sh --tier tier1 --cache off --reps 3 --prompt-tokens 512 --gen-tokens 256`
- Tier1 cache-on:
  `scripts/expert-cache/run-tier-bench.sh --tier tier1 --cache on --reps 3 --prompt-tokens 512 --gen-tokens 256`
- Dry-run command print only:
  `scripts/expert-cache/run-tier-bench.sh --tier tier2 --cache on --dry-run`
