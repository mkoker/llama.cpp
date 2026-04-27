# Tier 1 Correctness Gate — 2026-04-27

Command under test:
`LD_LIBRARY_PATH=$PWD/build-hip/bin ./build-hip/bin/llama-cli -m /mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf -ngl 99 -ncmoe 8 --expert-cache-size 8192 -p "Explain the Rayleigh-Jeans law in one paragraph." -n 256 -no-cnv`

## Build gate
Passed after HIP reconfigure and compile fix in `ggml/src/ggml-cuda/ggml-cuda.cu`.

## Test gate result
Blocked by model-load OOM on VM100 while production inference process retains VRAM.

Observed stderr excerpt:
- `--no-conversation is not supported by llama-cli`
- `allocating 14635.43 MiB on device 0: cudaMalloc failed: out of memory`
- `llama_model_load_from_file_impl: failed to load model`

Additional host state during gate:
- `rocm-smi` reports ~81% VRAM allocated with no KFD PIDs shown.
- `sudo rocm-smi -d 0 --gpureset` succeeds but reported allocation remains ~81%.

## Conclusion
Tier 1 correctness generation could not be completed with the exact gate command due to VRAM unavailability on VM100 at runtime.
