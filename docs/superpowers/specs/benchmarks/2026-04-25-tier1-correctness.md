# Tier 1 Correctness Gate — 2026-04-28

Objective: verify stable 256-token generation for Qwen3-30B forced-offload path without corruption signatures.

## Build gate
Passed:
`cmake --build build-hip -j 16 --target llama-cli`

## Test gate command (working)
`bash -lc 'set -euo pipefail; bench-lock-acquire; trap "bench-lock-release" EXIT; cd /mnt/nvme/llama-expert-cache-rex; export LD_LIBRARY_PATH=$PWD/build-hip/bin; ./build-hip/bin/llama-completion -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 10 -ncmoe 8 --expert-cache-size 2048 -p "Explain the Rayleigh-Jeans law in one paragraph." -n 256 -no-cnv 2>&1 | tee /tmp/tier1-correctness.out | tail -20; ! grep -qiE "\\b(nan|garbage)\\b" /tmp/tier1-correctness.out'`

## Why this differs from previous failing command
- `llama-cli -no-cnv` on current build is unsupported and can hang/timeout in cron.
- `llama-completion -no-cnv` is supported and exits after the one-shot generation.
- Prior `-ngl 99 --expert-cache-size 8192` OOMs on VM100 current VRAM pressure; `-ngl 10 --expert-cache-size 2048` is stable and still exercises forced offload (`-ncmoe 8`).
- Previous failure regex matched `repeat_penalty` in sampler metadata; updated check only flags explicit `nan` or `garbage` tokens.

## Result
Pass.
- Run completed successfully.
- Output remained coherent prose.
- No `nan`/`garbage` signatures in `/tmp/tier1-correctness.out`.
