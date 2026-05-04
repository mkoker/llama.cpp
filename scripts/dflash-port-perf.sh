#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 TARGET.gguf DFLASH_DRAFT.gguf" >&2
  exit 2
fi

TARGET=$1
DRAFT=$2
ROOT=${ROOT:-$(pwd)}
BIN_DIR="$ROOT/build-hip/bin"
CLI="$BIN_DIR/llama-cli"
SPEC="$BIN_DIR/llama-speculative"
RUNS=${RUNS:-5}
TOKENS=${TOKENS:-64}
DRAFT_N=${DRAFT_N:-16}
PROMPT=${PROMPT:-"Write a function to compute the nth Fibonacci number in Python."}
BASE_ARGS=(-m "$TARGET" -ngl 99 -p "$PROMPT" -n "$TOKENS" --temp 0 --seed 123 --no-warmup -no-cnv -st)
SPEC_ARGS=(-m "$TARGET" -md "$DRAFT" -ngl 99 -ngld 99 -p "$PROMPT" -n "$TOKENS" --temp 0 --seed 123 --draft "$DRAFT_N")

if [[ ! -x "$CLI" || ! -x "$SPEC" ]]; then
  echo "missing binaries under $BIN_DIR" >&2
  exit 2
fi
if [[ ! -f "$TARGET" || ! -f "$DRAFT" ]]; then
  echo "missing model(s): target=$TARGET draft=$DRAFT" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"
TMP=${TMPDIR:-/tmp}/dflash-perf-$$
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

run_one() {
  local name=$1; shift
  local out="$TMP/$name.out"
  "$@" >"$out.stdout" 2>"$out.stderr"
  cat "$out.stdout" "$out.stderr" >"$out"
  python3 - "$out" <<'PY'
import re, sys
text=open(sys.argv[1], errors='ignore').read()
# llama.cpp emits either "decoded ... speed: X t/s" or perf rows with "X tokens per second".
vals=[float(x) for x in re.findall(r'decoded\s+\d+\s+tokens\s+in\s+[0-9.]+\s+seconds,\s+speed:\s+([0-9.]+)\s+t/s', text)]
if not vals:
    vals=[float(x) for x in re.findall(r'Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s', text)]
if not vals:
    vals=[float(x) for x in re.findall(r'([0-9]+(?:\.[0-9]+)?)\s+tokens per second', text)]
if not vals:
    print('nan')
else:
    print(vals[-1])
PY
}

median() {
  python3 - "$@" <<'PY'
import math, statistics, sys
xs=[float(x) for x in sys.argv[1:] if x != 'nan' and math.isfinite(float(x)) and float(x) > 0]
if not xs:
    raise SystemExit(1)
print(f"{statistics.median(xs):.6f}")
PY
}

base=()
spec=()
for i in $(seq 1 "$RUNS"); do
  echo "run $i/$RUNS baseline" >&2
  base+=("$(run_one "base-$i" timeout 300s "$CLI" "${BASE_ARGS[@]}")")
  echo "run $i/$RUNS dflash" >&2
  spec+=("$(run_one "spec-$i" timeout 300s "$SPEC" "${SPEC_ARGS[@]}")")
done

base_med=$(median "${base[@]}")
spec_med=$(median "${spec[@]}")
speedup=$(python3 - "$base_med" "$spec_med" <<'PY'
import sys
b=float(sys.argv[1]); s=float(sys.argv[2])
print(f"{s/b:.6f}")
PY
)

last_spec="$TMP/spec-$RUNS.out"
n_drafted=$(awk '/n_drafted/{print $3}' "$last_spec" | tail -1)
n_accept=$(awk '/n_accept/{print $3}' "$last_spec" | tail -1)
accept=$(python3 - "${n_accept:-0}" "${n_drafted:-0}" <<'PY'
import sys
a=float(sys.argv[1] or 0); d=float(sys.argv[2] or 0)
print(f"{(a/d if d > 0 else 0.0):.6f}")
PY
)

echo "BASE_TPS $base_med"
echo "DFLASH_TPS $spec_med"
echo "SPEEDUP $speedup"
echo "ACCEPT $accept"

python3 - "$speedup" "$accept" <<'PY'
import sys
sp=float(sys.argv[1]); acc=float(sys.argv[2])
raise SystemExit(0 if sp >= 1.3 and acc >= 0.5 else 1)
PY
