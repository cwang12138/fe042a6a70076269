#!/bin/bash

set -e

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${script_dir}"

run_task() {
  local label="$1"
  shift

  if [ "$#" -eq 0 ]; then
    set -- bash "${label}"
  fi

  echo "==== Running ${label} ===="
  "$@"
}

# MegaCRN + STBoost on the two bundled datasets.
run_task scripts/MegaCRN/enhance/PEMS03.sh
run_task scripts/MegaCRN/enhance/PEMS08.sh
echo "All tasks completed."
