#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/mnt/nvme/perf-tracker"
CFG_PATH="${ROOT_DIR}/config.yaml"
RESULTS_DIR="${ROOT_DIR}/results"
SITE_DIR="${ROOT_DIR}/site"
BUILD_PAGE_PY="${ROOT_DIR}/build_page.py"

usage() {
  cat <<USAGE
Usage: $0 [--dry-run]

Options:
  --dry-run   Print what would run, do not execute benchmarks
  -h, --help  Show this help
USAGE
}

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "missing config: ${CFG_PATH}" >&2
  exit 1
fi

mkdir -p "${RESULTS_DIR}" "${SITE_DIR}"
MONTH_FILE="${RESULTS_DIR}/$(date -u +%Y-%m).jsonl"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if command -v bench-lock-acquire >/dev/null 2>/dev/null; then
    bench-lock-acquire
    cleanup_lock() {
      bench-lock-release || true
    }
    trap cleanup_lock EXIT
  fi
fi

mapfile -t RUN_MATRIX < <(
  python3 - <<'PYMATRIX'
import os
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path('/mnt/nvme/perf-tracker/config.yaml').read_text())
models = cfg.get('models', [])
backends = cfg.get('backends', [])

if os.getenv('SMOKE') == '1':
    models = models[:1]
    backends = backends[:1]

for m in models:
    name = str(m.get('name', '')).strip()
    model = str(m.get('model', '')).strip()
    n_ctx = int(m.get('n_ctx', 0) or 0)
    for b in backends:
        backend = str(b).strip()
        if name and model and backend:
            print(f"{name}|{model}|{n_ctx}|{backend}")
PYMATRIX
)

if [[ "${#RUN_MATRIX[@]}" -eq 0 ]]; then
  echo "no model/backend combinations resolved from config" >&2
  exit 1
fi

for row in "${RUN_MATRIX[@]}"; do
  IFS='|' read -r MODEL_NAME MODEL_FILE N_CTX BACKEND <<<"${row}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "would run backend=${BACKEND} model=${MODEL_NAME} file=${MODEL_FILE} n_ctx=${N_CTX}"
    continue
  fi

  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  RECORD_JSON=$(python3 - <<PYREC
import json
record = {
  "ts": "${TS}",
  "model_name": "${MODEL_NAME}",
  "model_file": "${MODEL_FILE}",
  "backend": "${BACKEND}",
  "n_ctx": int("${N_CTX}"),
  "status": "queued"
}
print(json.dumps(record, separators=(",",":")))
PYREC
)

  echo "${RECORD_JSON}" >> "${MONTH_FILE}"
done

if [[ "${DRY_RUN}" -eq 0 && -f "${BUILD_PAGE_PY}" ]]; then
  python3 "${BUILD_PAGE_PY}" --input-dir "${RESULTS_DIR}" --output "${SITE_DIR}/index.html" >/dev/null 2>/dev/null || true
fi

echo "ok"