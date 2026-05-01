#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert a Hugging Face DFlashDraftModel checkpoint to GGUF.

This is an intentionally small, architecture-specific draft converter for
z-lab DFlash drafter checkpoints. It preserves DFlash tensor names and writes
DFlash-specific metadata so the llama.cpp loader work in follow-up tasks has a
stable file format to consume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if "NO_LOCAL_GGUF" not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parent / "gguf-py"))

import gguf  # noqa: E402
from gguf.utility import SafetensorsLocal  # noqa: E402

LOGGER = logging.getLogger("convert-dflash-draft")
ARCH = "dflashdraft"
DEFAULT_OUTTYPE = "bf16"

_DTYPE_MAP: dict[str, tuple[np.dtype[Any], gguf.GGMLQuantizationType | None]] = {
    "F64": (np.dtype(np.float64), None),
    "F32": (np.dtype(np.float32), None),
    "F16": (np.dtype(np.float16), None),
    # NumPy has no native stable bfloat16 dtype. Store BF16 as raw bytes and
    # tell GGUF the logical tensor type/shape via raw_dtype/raw_shape.
    "BF16": (np.dtype(np.uint8), gguf.GGMLQuantizationType.BF16),
    "I64": (np.dtype(np.int64), None),
    "I32": (np.dtype(np.int32), None),
    "I16": (np.dtype(np.int16), None),
    "I8": (np.dtype(np.int8), None),
    "U8": (np.dtype(np.uint8), None),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_weight_files(model_dir: Path) -> list[Path]:
    safetensors = sorted(model_dir.glob("*.safetensors"))
    if safetensors:
        index = model_dir / "model.safetensors.index.json"
        if index.is_file():
            weight_map = load_json(index).get("weight_map", {})
            ordered = []
            seen = set()
            for filename in weight_map.values():
                if filename not in seen:
                    seen.add(filename)
                    ordered.append(model_dir / filename)
            return ordered
        return safetensors

    bins = sorted(model_dir.glob("pytorch_model*.bin"))
    if bins:
        return bins

    raise FileNotFoundError(f"no .safetensors or pytorch_model*.bin files found in {model_dir}")


def iter_safetensors(path: Path) -> Iterable[tuple[str, np.ndarray, tuple[int, ...], gguf.GGMLQuantizationType | None]]:
    with SafetensorsLocal(path) as tensors:
        for name, local in tensors.items():
            np_dtype, raw_dtype = _DTYPE_MAP.get(local.dtype, (None, None))  # type: ignore[assignment]
            if np_dtype is None:
                raise ValueError(f"unsupported safetensors dtype {local.dtype!r} for tensor {name!r}")

            if raw_dtype == gguf.GGMLQuantizationType.BF16:
                # Shape is byte-shaped on disk for add_tensor(raw_dtype=BF16),
                # with the last logical dimension expanded by 2 bytes.
                raw_shape = (*local.shape[:-1], local.shape[-1] * 2) if local.shape else (local.data_range.size,)
                arr = np.memmap(local.data_range.filename, mode="c", dtype=np.uint8,
                                offset=local.data_range.offset, shape=raw_shape)
            else:
                arr = np.memmap(local.data_range.filename, mode="c", dtype=np_dtype,
                                offset=local.data_range.offset, shape=local.shape)
                raw_shape = local.shape
            yield name, arr, raw_shape, raw_dtype


def iter_bin(path: Path) -> Iterable[tuple[str, np.ndarray, tuple[int, ...], gguf.GGMLQuantizationType | None]]:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("converting pytorch_model*.bin requires torch; prefer safetensors checkpoints") from exc

    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for name, tensor in state.items():
        if not hasattr(tensor, "detach"):
            continue
        tensor = tensor.detach().cpu()
        if tensor.dtype is torch.bfloat16:
            arr = tensor.view(torch.uint16).numpy().view(np.uint8).reshape(*tensor.shape[:-1], tensor.shape[-1] * 2)
            yield name, arr, arr.shape, gguf.GGMLQuantizationType.BF16
        else:
            yield name, tensor.numpy(), tuple(tensor.shape), None


def normalize_tensor_name(name: str) -> str:
    """Keep names close to HF while making the file visibly DFlash-specific."""
    if name.startswith("model."):
        name = name[len("model."):]
    return f"dflash.{name}"


def add_metadata(writer: gguf.GGUFWriter, config: dict[str, Any], model_dir: Path, outtype: str) -> None:
    dflash_config = config.get("dflash_config") or {}
    target_layer_ids = dflash_config.get("target_layer_ids") or config.get("target_layer_ids") or []

    writer.add_name(config.get("_name_or_path") or model_dir.name)
    writer.add_description("DFlash draft model converted from Hugging Face format")
    writer.add_file_type({"f32": 0, "f16": 1, "bf16": 32}[outtype])

    # Standard llama.cpp-style keys under the new architecture prefix.
    if "max_position_embeddings" in config:
        writer.add_context_length(int(config["max_position_embeddings"]))
    if "hidden_size" in config:
        writer.add_embedding_length(int(config["hidden_size"]))
    if "num_hidden_layers" in config:
        writer.add_block_count(int(config["num_hidden_layers"]))
    if "intermediate_size" in config:
        writer.add_feed_forward_length(int(config["intermediate_size"]))
    if "num_attention_heads" in config:
        writer.add_head_count(int(config["num_attention_heads"]))
    if "num_key_value_heads" in config:
        writer.add_head_count_kv(int(config["num_key_value_heads"]))
    if "rms_norm_eps" in config:
        writer.add_layer_norm_rms_eps(float(config["rms_norm_eps"]))

    # DFlash-specific extension keys. Follow-up llama.cpp tasks can either keep
    # these string keys or promote them to typed LLM_KV constants.
    writer.add_string(f"{ARCH}.source_hf_architecture", ",".join(config.get("architectures", [])))
    writer.add_uint32(f"{ARCH}.block_size", int(config.get("block_size", dflash_config.get("block_size", 16))))
    if "mask_token_id" in dflash_config or "mask_token_id" in config:
        writer.add_uint32(f"{ARCH}.mask_token_id", int(dflash_config.get("mask_token_id", config.get("mask_token_id"))))
    if "num_target_layers" in config:
        writer.add_uint32(f"{ARCH}.target_layer_count", int(config["num_target_layers"]))
    if target_layer_ids:
        writer.add_array(f"{ARCH}.target_layer_ids", [int(x) for x in target_layer_ids])


def convert(model_dir: Path, outfile: Path, outtype: str, *, dry_run: bool = False) -> int:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {model_dir}")
    config = load_json(config_path)

    archs = {str(x).lower() for x in config.get("architectures", [])}
    if archs and "dflash" not in " ".join(archs):
        LOGGER.warning("config architectures do not mention DFlash: %s", sorted(archs))

    weight_files = find_weight_files(model_dir)
    writer = gguf.GGUFWriter(path=None, arch=ARCH)
    add_metadata(writer, config, model_dir, outtype)

    tensor_count = 0
    seen: set[str] = set()
    for weight_file in weight_files:
        LOGGER.info("reading %s", weight_file.name)
        iterator = iter_safetensors(weight_file) if weight_file.suffix == ".safetensors" else iter_bin(weight_file)
        for src_name, tensor, raw_shape, raw_dtype in iterator:
            dst_name = normalize_tensor_name(src_name)
            if dst_name in seen:
                raise ValueError(f"duplicate tensor after normalization: {dst_name}")
            seen.add(dst_name)
            writer.add_tensor(dst_name, tensor, raw_shape=raw_shape, raw_dtype=raw_dtype)
            tensor_count += 1

    if dry_run:
        print(f"DFlash dry-run OK: tensors={tensor_count} outfile={outfile}")
        return tensor_count

    outfile.parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file(outfile)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"Wrote DFlash GGUF: {outfile} ({tensor_count} tensors)")
    return tensor_count


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a z-lab DFlash draft Hugging Face checkpoint to GGUF.",
    )
    parser.add_argument("model", type=Path, nargs="?", help="Path to DFlashDraftModel HF directory")
    parser.add_argument("--outfile", "-o", type=Path, help="Output GGUF file")
    parser.add_argument("--outtype", choices=("f32", "f16", "bf16"), default=DEFAULT_OUTTYPE,
                        help="GGUF file type metadata; tensors are preserved in checkpoint dtype where possible")
    parser.add_argument("--dry-run", action="store_true", help="Parse config/tensors and report count without writing GGUF")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    if args.model is None:
        parser.error("model directory is required unless using --help")
    model_dir = args.model.expanduser().resolve()
    outfile = args.outfile or model_dir.with_suffix(".gguf")
    convert(model_dir, outfile.expanduser().resolve(), args.outtype, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
