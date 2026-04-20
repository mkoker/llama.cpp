#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage:
  $0 --tier tier1|tier2|tier3 --cache on|off [--reps N] [--prompt-tokens N] [--gen-tokens N] [--ncmoe N] [--dry-run]

Notes:
  - tier1 uses forced offload path with -ncmoe 8 for cache efficacy checks.
  - set GGML_EXPERT_CACHE_DEBUG=1 for copy-path counters in logs.
USAGE
}

TIER=""
CACHE=""
REPS=3
PROMPT_TOKENS=512
GEN_TOKENS=256
NCMOE_OVERRIDE=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="${2:-}"; shift 2 ;;
    --cache) CACHE="${2:-}"; shift 2 ;;
    --reps) REPS="${2:-}"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="${2:-}"; shift 2 ;;
    --gen-tokens) GEN_TOKENS="${2:-}"; shift 2 ;;
    --ncmoe) NCMOE_OVERRIDE="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$TIER" ]] || { echo "--tier required" >&2; usage; exit 2; }
[[ "$CACHE" == "on" || "$CACHE" == "off" ]] || { echo "--cache must be on|off" >&2; usage; exit 2; }

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BIN="$ROOT/build-hip/bin/llama-bench"
LD_PATH="$ROOT/build-hip/bin"

case "$TIER" in
  tier1)
    MODEL="/mnt/nvme/models/Qwen3-30B-A3B-Q4_K_M.gguf"
    NCMOE=8
    NGL=99
    ;;
  tier2)
    MODEL="/mnt/nvme/MiniMax-M2.7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
    NCMOE=8
    NGL=10
    ;;
  tier3)
    MODEL="/mnt/nvme/models/qwen3-235b/Qwen3-235B-A22B-Q4_K_M-00001-of-00005.gguf"
    NCMOE=8
    NGL=10
    ;;
  *)
    echo "Invalid tier: $TIER" >&2
    exit 2
    ;;
esac

[[ -n "$NCMOE_OVERRIDE" ]] && NCMOE="$NCMOE_OVERRIDE"

CACHE_ARGS=()
if [[ "$CACHE" == "on" ]]; then
  CACHE_ARGS+=(--expert-cache-size 2048)
fi

COMMON_ARGS=(
  -m "$MODEL"
  -ngl "$NGL"
  -b 1
  -t 1
  -p "$PROMPT_TOKENS"
  -n "$GEN_TOKENS"
  -ncmoe "$NCMOE"
  -fa 1
)

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="$ROOT/logs/expert-cache"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${TIMESTAMP}-${TIER}-cache-${CACHE}.log"

CMD=(env LD_LIBRARY_PATH="$LD_PATH" "$BIN" "${COMMON_ARGS[@]}" "${CACHE_ARGS[@]}")

{
  echo "timestamp=$TIMESTAMP"
  echo "tier=$TIER"
  echo "cache=$CACHE"
  echo "reps=$REPS"
  echo "prompt_tokens=$PROMPT_TOKENS"
  echo "gen_tokens=$GEN_TOKENS"
  printf "command="; printf "%q " "${CMD[@]}"; echo
} | tee "$LOG_FILE"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "dry_run=1" | tee -a "$LOG_FILE"
  exit 0
fi

if ! command -v bench-lock-acquire >/dev/null 2>&1; then
  echo "bench-lock-acquire not found" >&2
  exit 3
fi
if ! command -v bench-lock-release >/dev/null 2>&1; then
  echo "bench-lock-release not found" >&2
  exit 3
fi

cleanup() {
  bench-lock-release >/dev/null 2>&1 || true
}
trap cleanup EXIT

bench-lock-acquire

for ((i=1; i<=REPS; i++)); do
  echo "rep=$i/$REPS" | tee -a "$LOG_FILE"
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  echo "rep_done=$i" | tee -a "$LOG_FILE"
done

echo "log_file=$LOG_FILE"
