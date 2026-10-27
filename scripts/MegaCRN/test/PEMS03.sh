#!/bin/bash

set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../../.."

for q_len in ${STB_Q_LIST:-12 24 48 96}; do
  case "$q_len" in
    12) run_id='2026-10-16-12-23-23-686122a8' ;;
    24) run_id='2026-10-16-12-36-20-6f72fb89' ;;
    48) run_id='2026-10-13-12-41-39-ed09cbba' ;;
    96) run_id='2026-10-13-12-50-50-f072df52' ;;
    *) echo "Unsupported horizon: $q_len" >&2; exit 1 ;;
  esac
  "${PYTHON_BIN:-python}" -u main.py \
    --test_only 1 \
    --best_model_path "./experiments/MegaCRN_True/PEMS03/96-${q_len}/${run_id}/best_model.pt" \
    --gpu "${GPU:-0}" "$@"
done
echo "PEMS03 checkpoint tests completed."
