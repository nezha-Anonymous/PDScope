#!/usr/bin/env bash
set -euo pipefail
# Build the native layer dispatch used by the default decode path: it submits the
# packed CPU expert group, joins it, stages the rows and merges them with one
# weighted scatter. ATen + cudart only, so no nvcc is involved.
root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON_BIN:-python}
build_dir=${PDSCOPE_TAIL_BUILD_DIR:-"$root_dir/.cache/pdscope-native/build-layer-dispatch"}
out_dir=${PDSCOPE_TAIL_OUTPUT_DIR:-"$root_dir/expertbench/native-build/layer-dispatch"}
mkdir -p "$build_dir" "$out_dir"
"$python_bin" - "$root_dir" "$build_dir" "$out_dir" <<'PY'
import hashlib, json, pathlib, shutil, sys
from torch.utils.cpp_extension import load
root, build_dir, out_dir = (pathlib.Path(p).resolve() for p in sys.argv[1:4])
source = root / 'native' / 'native_layer_dispatch.cpp'
module = load(name='pdscope_layer_dispatch', sources=[str(source)],
              build_directory=str(build_dir), verbose=False, with_cuda=True,
              extra_cflags=['-O2'], extra_cuda_cflags=['-O2'])
path = pathlib.Path(module.__file__).resolve()
target = out_dir / path.name
shutil.copyfile(path, target)
record = dict(name='pdscope_layer_dispatch', path=str(target),
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              entry="submit_group+finish_and_merge", scope='one H2D copy plus one weighted scatter per layer',
              requires_cuda_toolchain=False)
(out_dir / 'variant.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
PY
