#!/bin/bash

set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}/../../.."

for q_len in ${STB_Q_LIST:-12 24 48 96}; do
  case "$q_len" in
    12) run_id='2026-10-16-15-11-22-40fb9551' ;;
    24) run_id='2026-10-16-15-16-02-7afa073f' ;;
    48) run_id='2026-10-13-15-25-58-e085ea85' ;;
    96) run_id='2026-10-13-15-32-01-74e463c5' ;;
    *) echo "Unsupported horizon: $q_len" >&2; exit 1 ;;
  esac
  "${PYTHON_BIN:-python}" -u main.py \
    --test_only 1 \
    --best_model_path "./experiments/MegaCRN_True/PEMS08/96-${q_len}/${run_id}/best_model.pt" \
    --gpu "${GPU:-0}" "$@"
done
echo "PEMS08 checkpoint tests completed."
