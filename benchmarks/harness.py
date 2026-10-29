"""Shared measurement harness: manifests, subprocess capture and the two-layer
numerical acceptance helpers used by ``benchmarks/pdscope_compare.py``.

It is deliberately small: what a run must record to be reproducible, and the
numerical checks the two-layer mode asserts.
"""
from __future__ import annotations

import ast
import contextlib
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def digest(payload):
    """SHA-256 of a file path or of raw bytes."""
    if isinstance(payload, (bytes, bytearray)):
        return hashlib.sha256(payload).hexdigest()
    return hashlib.sha256(Path(payload).read_bytes()).hexdigest()


def error_metrics(actual, reference):
    import torch
    delta = (actual.float().cpu() - reference.float().cpu()).reshape(-1)
    ref = reference.float().cpu().reshape(-1)
    norm = torch.linalg.vector_norm(ref).item()
    rms = torch.sqrt(torch.mean(ref * ref)).item()
    return dict(relative_l2=torch.linalg.vector_norm(delta).item() / max(norm, 1e-12),
                max_abs_over_reference_rms=delta.abs().max().item() / max(rms, 1e-12),
                max_abs=delta.abs().max().item(), reference_rms=rms,
                finite=bool(torch.isfinite(actual).all() and torch.isfinite(reference).all()))


def _difference_location(actual, reference):
    import torch
    a, b = actual.float().cpu().reshape(-1), reference.float().cpu().reshape(-1)
    position = int((a-b).abs().argmax())
    return dict(flat_index=position, native_value=a[position].item(), torch_value=b[position].item(),
                difference=(a[position]-b[position]).item())


def paired_prefill_diagnostic(inputs, routing, masters, layer, placements, cpu_backend, actual):
    """Independent one-hot dispatch; compare queued native math to serial math.

    FP32 reference is reported alongside both BF16 implementations. This is a
    targeted arithmetic/scheduling check, not model/task accuracy evaluation.
    """
    import torch
    from expertbench.adapters.qwen import reference_fp32
    from expertbench.cache import ExpertKey
    experts = len(masters)
    mask = torch.nn.functional.one_hot(routing.selected, num_classes=experts).permute(2, 1, 0)
    serial = torch.zeros_like(inputs)
    fp32 = torch.zeros_like(inputs, device='cpu', dtype=torch.float32)
    native_torch = torch.zeros_like(inputs)
    cpu_calls = 0
    with torch.inference_mode():
        for expert in range(experts):
            slots, tokens = torch.where(mask[expert])
            if not len(tokens):
                continue
            key = ExpertKey(layer, expert, 'bf16')
            x = inputs[tokens]
            coeff = routing.weights[tokens, slots, None]
            if placements[expert] == 'cpu':
                part = cpu_backend(key, x.cpu()).to(inputs.device)
                torch_part, _ = native_expert_reference(masters[key], x.cpu())
                torch_part = torch_part.to(inputs.device)
                cpu_calls += 1
            else:
                part, _ = native_expert_reference(masters[key], x)
                torch_part = part
            serial.index_add_(0, tokens, (part * coeff).to(inputs.dtype))
            native_torch.index_add_(0, tokens, (torch_part * coeff).to(inputs.dtype))
            precise = reference_fp32(x.cpu(), masters[key].tensors)
            fp32.index_add_(0, tokens.cpu(), precise * coeff.float().cpu())
    return dict(async_vs_serial_bitwise=bool(torch.equal(actual, serial)),
                async_vs_serial=error_metrics(actual, serial),
                native_queue_vs_fp32=error_metrics(actual, fp32),
                torch_control_vs_fp32=error_metrics(native_torch, fp32),
                reference_native_calls=cpu_calls,
                max_difference_location=_difference_location(actual, native_torch))


def native_expert_reference(weights, inputs):
    """Installed original MLP forward, with selected checkpoint storage bound."""
    import torch
    path = Path(importlib.metadata.distribution("transformers").locate_file(
        "transformers/models/qwen3_moe/modeling_qwen3_moe.py"))
    source = path.read_text()
    definitions = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "Qwen3MoeMLP"]
    if len(definitions) != 1:
        raise ValueError("native expert reference class unavailable")
    namespace = {"torch": torch, "nn": torch.nn, "ACT2FN": {"silu": torch.nn.functional.silu}}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    intermediate, hidden = weights.tensors["gate_proj"].shape
    from types import SimpleNamespace
    config = SimpleNamespace(hidden_size=hidden, intermediate_size=intermediate, hidden_act="silu")
    with torch.device("meta"):
        module = namespace["Qwen3MoeMLP"](config, intermediate_size=intermediate)
    for name, tensor in weights.tensors.items():
        getattr(module, name).weight = torch.nn.Parameter(tensor.to(inputs.device), requires_grad=False)
    with torch.inference_mode():
        actual = module(inputs)
    return actual, {"source": str(path), "sha256": digest(source.encode()),
                    "class": "Qwen3MoeMLP", "forward_modified": False}


def command(argv):
    proc = subprocess.run(argv, text=True, capture_output=True, timeout=30)
    return dict(argv=argv, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


def capture_manifest(args, root):
    import importlib.metadata
    packages = {}
    for name in ('torch', 'transformers', 'safetensors'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    paths = subprocess.check_output(["git", "ls-files", "-z", "--", "*.py"], cwd=root).split(b"\0")
    # The native extension sources and their build recipes are included even
    # while untracked/dirty: a result must be reproducible from archived source
    # bytes, not from an ignored .so left on the node.
    explicit = ("native/native_layer_dispatch.cpp", "native/build_native_layer_dispatch.sh",
                "native/build_kt_kernel.sh", "native/build_kt_kernel_avx512.sh")
    paths = list(paths) + [name.encode() for name in explicit if (root / name).is_file()]
    # The index may still list files deleted in this worktree; archive only bytes
    # that are actually present, so the snapshot is reproducible from what ran.
    paths = sorted({raw for raw in paths if raw and (root / raw.decode()).is_file()})
    members = {}
    archive = args.output.parent / "source.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for raw in paths:
            if not raw:
                continue
            relative = raw.decode()
            content = (root / relative).read_bytes()
            z.writestr(relative, content)
            members[relative] = hashlib.sha256(content).hexdigest()
    native_dir = root / "expertbench" / "native-build"
    native_artifacts = (sorted(native_dir.glob("_native_prefetch*.so"))
                        + sorted((native_dir / "kt-prefetch").glob("kt_kernel_ext*.so"))) if native_dir.is_dir() else []
    native_source = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in explicit if (root / name).is_file()
    }
    native_build = dict(source_hashes=native_source, artifact_count=len(native_artifacts),
                        artifact_roles=['companion_source_ranges', 'kt_internal_BufferB_variant'])
    if native_artifacts:
        native_build["artifacts"] = [dict(path=str(path.resolve()),
                                           role=('companion_source_ranges' if path.name.startswith('_native_prefetch')
                                                 else 'kt_internal_BufferB_variant'),
                                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                           bytes=path.stat().st_size)
                                     for path in native_artifacts]
    return dict(schema_version=1, entry="benchmarks/pdscope_compare.py", argv=sys.argv,
                config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), cwd=os.getcwd(),
                python=sys.version, executable=sys.executable, host=platform.node(), platform=platform.platform(),
                package_versions=packages,
                process_argv=Path('/proc/self/cmdline').read_bytes().decode().split('\0')[:-1],
                affinity=sorted(os.sched_getaffinity(0)), env={key: os.environ.get(key) for key in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME",
                     "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "PYTHONHASHSEED")},
                git_commit=command(["git", "-C", str(root), "rev-parse", "HEAD"]),
                git_status=command(["git", "-C", str(root), "status", "--short", "--", "."]),
                nvidia_smi=command(["nvidia-smi"]), lscpu=command(["lscpu", "-J"]),
                source_archive=dict(path=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                    members=members, includes_dirty_worktree_bytes=True),
                native_prefetch_build=native_build, status="running")


@contextlib.contextmanager
def capture_native_stdio(path):
    """Capture C/C++ stdio, which Python's redirect_stdout does not redirect."""
    sys.stdout.flush();sys.stderr.flush()
    libc=ctypes.CDLL(None)
    libc.fflush(None)
    saved=[os.dup(fd) for fd in (1,2)]
    try:
        with Path(path).open('wb',buffering=0) as stream:
            try:
                os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
                yield
            finally:
                libc.fflush(None)
                for fd,previous in zip((1,2),saved,strict=True):os.dup2(previous,fd)
    finally:
        for fd in saved:os.close(fd)
