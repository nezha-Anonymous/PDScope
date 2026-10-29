#!/usr/bin/env bash
# Build everything this repository needs, using the host toolchain:
#   1. the native layer dispatch (one call for the CPU expert group's tail)
#   2. the CPU expert kernel   (pinned ktransformers, AVX2 by default)
#
# Nothing here needs nvcc: the extension uses only ATen and the CUDA runtime.
set -euo pipefail
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHON_BIN=${PYTHON_BIN:-python}

echo "== 1/2 native layer dispatch"
"$root_dir/native/build_native_layer_dispatch.sh"

echo "== 2/2 CPU expert kernel"
if [ -z "${KT_SOURCE_ROOT:-}" ]; then
  cat >&2 <<'MSG'
KT_SOURCE_ROOT is not set. Fetch the pinned kernel source first:

  git clone https://github.com/kvcache-ai/ktransformers.git "$HOME/kt-src"
  git -C "$HOME/kt-src" checkout a02efe36b5802e7ca79a6b29146e01cf3b97d305
  git -C "$HOME/kt-src" submodule update --init --recursive   # llama.cpp + pybind11
  export KT_SOURCE_ROOT="$HOME/kt-src"

MSG
  exit 2
fi
KT_SOURCE_ROOT="$KT_SOURCE_ROOT" "$root_dir/native/build_kt_kernel.sh"
if [ "${WITH_AVX512:-0}" = "1" ]; then
  KT_SOURCE_ROOT="$KT_SOURCE_ROOT" "$root_dir/native/build_kt_kernel_avx512.sh"
fi

echo "== done"
ls -1 "$root_dir/expertbench/native-build"/*/*.so
