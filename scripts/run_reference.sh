#!/usr/bin/env bash
# Reference measurement: prefill and decode throughput at 512/1024/2048/4096
# input tokens. Usage:
#   ./scripts/run_reference.sh                 # writes runs/<UTC timestamp>/
#   ./scripts/run_reference.sh results/run1    # writes results/run1/result.json
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${PDSCOPE_MODEL:?export PDSCOPE_MODEL=/path/to/Qwen3-30B-A3B}"
: "${PDSCOPE_DATASET:?export PDSCOPE_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json}"
cd "$root_dir"
kernel=$(ls expertbench/native-build/kt-kernel/kt_kernel_ext*.so 2>/dev/null | head -n1)
[ -n "$kernel" ] || { echo "build the CPU expert kernel first: ./scripts/build_native.sh" >&2; exit 2; }

args=()
if [ $# -ge 1 ]; then
  args+=(--output "${1%/}/result.json")
fi

PYTHONPATH=. python benchmarks/pdscope_compare.py \
  --lengths 512 1024 2048 4096 --repetitions 2 \
  --pdscope-cpu-threads 18 --pdscope-decode-cpu-targets 6 \
  --cpu-extension "$kernel" \
  "${args[@]}"
