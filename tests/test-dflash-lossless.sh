#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
BIN="$ROOT/build-hip/bin/llama-speculative"
MODEL=${MODEL:-/mnt/nvme/models/Qwen3.5-0.8B-Q8_0.gguf}
OUT=${OUT:-/tmp/dflash-lossless.out}
ERR=${ERR:-/tmp/dflash-lossless.err}

if [[ ! -x "$BIN" ]]; then
  echo "missing llama-speculative: $BIN" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "missing lossless test model: $MODEL" >&2
  exit 1
fi

PROMPT="The capital of France is"
rm -f "$OUT" "$ERR"
timeout 180s env LD_LIBRARY_PATH="$ROOT/build-hip/bin:${LD_LIBRARY_PATH:-}" \
  "$BIN" \
    -m "$MODEL" -md "$MODEL" \
    -ngl 0 -ngld 0 \
    -p "$PROMPT" -n 4 \
    --temp 0 --seed 123 \
    --draft 4 --parallel 1 \
    >"$OUT" 2>"$ERR"

if ! grep -q "Paris" "$OUT"; then
  echo "lossless FAIL: expected generated text in stdout" >&2
  tail -20 "$OUT" >&2
  exit 1
fi

n_drafted=$(awk '/n_drafted/{print $3}' "$ERR" | tail -1)
n_accept=$(awk '/n_accept/{print $3}' "$ERR" | tail -1)
if [[ -z "${n_drafted:-}" || -z "${n_accept:-}" ]]; then
  echo "lossless FAIL: missing speculative counters" >&2
  tail -40 "$ERR" >&2
  exit 1
fi
if [[ "$n_drafted" -le 0 || "$n_accept" != "$n_drafted" ]]; then
  echo "lossless FAIL: accepted $n_accept of $n_drafted drafted tokens" >&2
  tail -40 "$ERR" >&2
  exit 1
fi

grep -q "n_decoded" "$ERR"
echo "lossless OK: accepted $n_accept/$n_drafted block-draft tokens"
