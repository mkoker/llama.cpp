# Regression safety check: dense Qwen3.6 cache-disabled

Task: Task 14 — Regression safety check on dense Qwen3.6 (cache disabled = no behavioral change).

Date: 2026-05-01T06:00Z
Workspace: `/mnt/nvme/llama-expert-cache-rex`
Branch: `rex/expert-cache-scheduler-v1`
Commit tested: `e2bbc30`

## Build gate

Command:

```bash
ssh ubuntu@192.168.1.169 'cd /mnt/nvme/llama-expert-cache-rex && cmake --build build-hip -j 16 --target llama-bench 2>&1 | tail -3'
```

Result:

```text
[ 89%] Built target llama
[ 98%] Built target common
[100%] Built target llama-bench
```

## Runtime gate

Command:

```bash
ssh ubuntu@192.168.1.169 'bench-lock-acquire && cd /mnt/nvme/llama-expert-cache-rex && LD_LIBRARY_PATH=$PWD/build-hip/bin ./build-hip/bin/llama-bench -m /mnt/nvme/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -fa 1 -p 0 -n 128 -r 1 2>&1 | tee /tmp/regression.out | tail -5; bench-lock-release; grep -q "tg128" /tmp/regression.out'
```

Result:

```text
acquired: 279282 2026-05-01T06:00:42+00:00 ubuntu@ai-server
| model                          |       size |     params | backend    | ngl | fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Medium |  20.60 GiB |    34.66 B | ROCm       |  99 |  1 |           tg128 |         71.62 ± 0.00 |

build: e2bbc30 (558)
released
```

## Verdict

Pass. With expert cache disabled (no `--expert-cache-size` flag), the HIP `llama-bench` dense/Qwen3.6 regression smoke completes and emits the expected `tg128` row. No cache-specific flag was used, so this validates the scheduler/cache wiring remains inert for the cache-disabled path under the gate workload.
