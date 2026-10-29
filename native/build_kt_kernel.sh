#!/usr/bin/env bash
set -euo pipefail

# Build the AVX2 BF16 expert kernel (the default) from the pinned ktransformers
# snapshot. Patch-free: it is the upstream operator, compiled for this host.
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source_root=${KT_SOURCE_ROOT:-"$root_dir/.cache/pdscope-native/ktransformers-a02efe36b5802e7ca79a6b29146e01cf3b97d305"}
build_dir=${KT_BUILD_DIR:-"$root_dir/.cache/pdscope-native/build-avx2"}
out_dir=${KT_OUTPUT_DIR:-"$root_dir/expertbench/native-build/kt-kernel"}
python_bin=${PYTHON_BIN:-python}
python_include=${PYTHON_INCLUDE_DIR:-$($python_bin -c "import sysconfig;print(sysconfig.get_paths()['include'])")}
python_library=${PYTHON_LIBRARY:-$($python_bin -c "import sysconfig;print(sysconfig.get_config_var('LIBPL')+'/libpython'+sysconfig.get_config_var('LDVERSION')+'.a')")}

test -d "$source_root/kt-kernel" || { echo "missing ktransformers source root: $source_root" >&2; exit 2; }
test -f "$source_root/third_party/llama.cpp/CMakeLists.txt" || {
  echo "missing llama.cpp submodule at $source_root/third_party/llama.cpp" >&2; exit 2; }
test -f "$source_root/third_party/pybind11/CMakeLists.txt" || {
  echo "missing pybind11 submodule at $source_root/third_party/pybind11" >&2; exit 2; }

cmake -S "$source_root/kt-kernel" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-DPDSCOPE_AVX2_ONLY=1 \
  -DKTRANSFORMERS_USE_CUDA=OFF -DKTRANSFORMERS_CPU_USE_AMX_AVX512=OFF \
  -DKTRANSFORMERS_CPU_USE_AMX=OFF -DKTRANSFORMERS_CPU_USE_KML=OFF \
  -DKTRANSFORMERS_CPU_USE_ONEDNN_VNNI=OFF -DLLAMA_NATIVE=OFF \
  -DLLAMA_FMA=ON -DLLAMA_F16C=ON -DLLAMA_AVX=ON -DLLAMA_AVX2=ON \
  -DLLAMA_AVX512=OFF -DLLAMA_AVX512_FANCY_SIMD=OFF \
  -DLLAMA_AVX512_BF16=OFF -DLLAMA_AVX512_VNNI=OFF -DLLAMA_AVX512_VBMI=OFF \
  -DPYTHON_EXECUTABLE="$python_bin" -DPYTHON_INCLUDE_DIR="$python_include" \
  -DPYTHON_LIBRARY="$python_library"
cmake --build "$build_dir" --target kt_kernel_ext -j"${CMAKE_BUILD_PARALLEL_LEVEL:-8}"

artifact=$(find "$build_dir" -maxdepth 1 -name 'kt_kernel_ext*.so' -type f | head -n1)
test -n "$artifact" || { echo "kernel artifact not found" >&2; exit 3; }
mkdir -p "$out_dir"
cp -f "$artifact" "$out_dir/$(basename "$artifact")"

"$python_bin" - "$out_dir/$(basename "$artifact")" "$source_root" <<'PY'
import hashlib, importlib.util, json, pathlib, subprocess, sys
artifact = pathlib.Path(sys.argv[1]).resolve()
source = pathlib.Path(sys.argv[2]).resolve()
spec = importlib.util.spec_from_file_location('kt_kernel_ext', artifact)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def rev(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None
meta = dict(status='built', source_root=str(source),
            source_commit='a02efe36b5802e7ca79a6b29146e01cf3b97d305',
            llama_cpp_commit=rev(source/'third_party/llama.cpp'),
            pybind11_commit=rev(source/'third_party/pybind11'),
            artifact=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            artifact_bytes=artifact.stat().st_size,
            cpu_variant=getattr(module, '__cpu_variant__', None), operator='AVX2BF16_MOE')
(artifact.parent/'kernel.json').write_text(json.dumps(meta, indent=2)+'\n')
print(json.dumps(meta, indent=2))
PY
