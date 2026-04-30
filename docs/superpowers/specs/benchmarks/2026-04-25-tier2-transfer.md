# Tier 2 transfer benchmark — MiniMax-M2.7 UD-IQ4_XS

Date (UTC): 2026-04-30T00:13:58.175875+00:00
Repo: /mnt/nvme/llama-expert-cache-rex
Commit: 23b0d60

## Gate diagnosis
- Plan gate setting (-ngl 99) is not runnable on VM100 for MiniMax-M2.7 UD-IQ4_XS.
- Observed failure: allocating 102760.78 MiB ... cudaMalloc failed: out of memory.
- Transfer proof run used required-offload config: -ngl 1 -ncmoe 8.

## Commands
- cache-off baseline:
  ./build-hip/bin/llama-bench -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf -ngl 1 -fa 1 -ncmoe 8 --expert-cache-size 0 -p 0 -n 128 -r 1
- cache-on:
  ./build-hip/bin/llama-bench -m /mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf -ngl 1 -fa 1 -ncmoe 8 --expert-cache-size 16384 -p 0 -n 128 -r 1

## Results (tg128)
- off: | minimax-m2 230B.A10B IQ4_XS - 4.25 bpw | 100.96 GiB |   228.69 B | ROCm       |   1 |          8 |  1 |           tg128 |          2.37 ± 0.00 |
- on:  | minimax-m2 230B.A10B IQ4_XS - 4.25 bpw | 100.96 GiB |   228.69 B | ROCm       |   1 |          8 |  1 |           tg128 |          2.86 ± 0.00 |

## Throughput delta
- off: 2.37 t/s
- on:  2.86 t/s
- gain: 0.49 t/s (20.68%)

## Verdict
PASS: cache provides speedup over no-cache offload baseline for Tier 2 transfer scenario.
