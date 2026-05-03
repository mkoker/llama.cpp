#!/usr/bin/env python3
"""Generate compact DFlash numeric reference samples from the HF PyTorch safetensors file.

The C++ gate test compares these BF16 samples against the converted GGUF.  Keeping
only deterministic sparse samples avoids checking a ~900 MiB reference artifact
into git while still validating the converter did not numerically corrupt the
DFlash draft weights used by the llama.cpp graph.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

TENSORS = [
    "fc.weight",
    "hidden_norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight",
    "layers.0.self_attn.k_norm.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.7.input_layernorm.weight",
    "layers.7.self_attn.q_proj.weight",
    "layers.7.self_attn.k_proj.weight",
    "layers.7.self_attn.v_proj.weight",
    "layers.7.self_attn.o_proj.weight",
    "layers.7.mlp.gate_proj.weight",
    "layers.7.mlp.up_proj.weight",
    "layers.7.mlp.down_proj.weight",
    "norm.weight",
]


def load_safetensors(path: Path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    return header, data


def tensor_indices(n_elem: int, samples: int, rng: random.Random) -> list[int]:
    fixed = [0, 1, 2, 3, max(0, n_elem // 2 - 1), n_elem // 2, n_elem - 4, n_elem - 3, n_elem - 2, n_elem - 1]
    fixed = [i for i in fixed if 0 <= i < n_elem]
    need = max(0, samples - len(set(fixed)))
    extra = rng.sample(range(n_elem), min(need, n_elem)) if need else []
    return sorted(set(fixed + extra))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/mnt/nvme/models/Qwen3-Coder-Next-DFlash-hf/model.safetensors")
    ap.add_argument("--out", default="/mnt/nvme/dflash-refs/qwen3-coder-next-dflash-bf16.samples.bin")
    ap.add_argument("--samples-per-tensor", type=int, default=64)
    args = ap.parse_args()

    header, data = load_safetensors(Path(args.hf))
    rng = random.Random(0xD1A5F1A5)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for tensor in TENSORS:
        if tensor not in header:
            raise SystemExit(f"missing tensor in safetensors: {tensor}")
        meta = header[tensor]
        if meta["dtype"] != "BF16":
            raise SystemExit(f"expected BF16 tensor, got {tensor}: {meta['dtype']}")
        start, end = meta["data_offsets"]
        n_elem = (end - start) // 2
        idxs = tensor_indices(n_elem, args.samples_per_tensor, rng)
        vals = [struct.unpack_from("<H", data, start + i * 2)[0] for i in idxs]
        records.append(("dflash." + tensor, idxs, vals))

    with out_path.open("wb") as f:
        f.write(b"DFLREF1\0")
        f.write(struct.pack("<I", len(records)))
        for name, idxs, vals in records:
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", len(idxs)))
            for idx, val in zip(idxs, vals):
                f.write(struct.pack("<IH", idx, val))

    print(f"wrote {out_path} tensors={len(records)} samples={sum(len(r[1]) for r in records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
