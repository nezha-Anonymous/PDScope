"""Selected Qwen3 MoE BF16 expert weights, without loading the full model."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from expertbench.cache import ExpertKey


def digest(payload):
    """SHA-256 of bytes or text, as recorded in every provenance block."""
    return hashlib.sha256(payload.encode() if isinstance(payload, str) else payload).hexdigest()


def read_json_bounded(path, cap_bytes=8 * 1024**2):
    """Read a small JSON document with an explicit size bound."""
    path = Path(path)
    if path.stat().st_size > cap_bytes:
        raise ValueError(f'{path} exceeds the {cap_bytes} byte JSON bound')
    return json.loads(path.read_text())


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def tensor_hash(tensor):
    return digest(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


@dataclass
class ExpertWeights:
    key: ExpertKey
    tensors: dict
    provenance: dict

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def hashes(self):
        return {name: tensor_hash(tensor) for name, tensor in self.tensors.items()}


def load_expert(model_path, key, max_bytes=64 * 1024**2):
    root = Path(model_path).resolve()
    config_path, index_path = root / "config.json", root / "model.safetensors.index.json"
    cfg, index = read_json_bounded(config_path), read_json_bounded(index_path)
    if cfg.get("model_type") != "qwen3_moe" or cfg.get("hidden_act") != "silu" or key.representation != "bf16":
        raise ValueError("only Qwen3 MoE SiLU with original BF16 experts is supported")
    if not 0 <= key.layer < cfg["num_hidden_layers"] or not 0 <= key.expert < cfg["num_experts"]:
        raise ValueError("expert index outside checkpoint config")
    hidden, intermediate = cfg["hidden_size"], cfg["moe_intermediate_size"]
    shapes = {"gate_proj": [intermediate, hidden], "up_proj": [intermediate, hidden],
              "down_proj": [hidden, intermediate]}
    expected_bytes = 3 * intermediate * hidden * 2
    if type(max_bytes) is not int or not 0 < expected_bytes <= max_bytes:
        raise ValueError("selected expert exceeds the explicit weight read budget")
    tensors, files = {}, {}
    for projection in PROJECTIONS:
        tensor_name = f"model.layers.{key.layer}.mlp.experts.{key.expert}.{projection}.weight"
        relative = Path(index["weight_map"][tensor_name])
        path = (root / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or root not in path.parents or path.suffix != ".safetensors":
            raise ValueError("unsafe or unsupported checkpoint shard path")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            view = handle.get_slice(tensor_name)
            if view.get_shape() != shapes[projection] or view.get_dtype() != "BF16":
                raise ValueError(f"checkpoint shape/dtype mismatch: {tensor_name}")
            # Own the selected CPU bytes after closing the shard mmap; no .to()
            # on a model module and no implicit conversion/quantization.
            tensors[projection] = handle.get_tensor(tensor_name).clone().contiguous()
        files[projection] = {"tensor_name": tensor_name, "shard": str(path),
                             "shard_bytes": path.stat().st_size, "shard_mtime_ns": path.stat().st_mtime_ns,
                             "shape": shapes[projection], "dtype": "bfloat16"}
    weights = ExpertWeights(key, tensors, {})
    weights.provenance.update(model_path=str(root), config_sha256=digest(config_path.read_bytes()),
                              index_sha256=digest(index_path.read_bytes()), tensors=files,
                              tensor_sha256=weights.hashes(), selected_bytes=weights.nbytes,
                              full_checkpoint_checksums="not_computed", checkpoint_revision="unverified")
    return weights


def expert_forward(inputs, tensors):
    """Native expert formula, with no router, attention or implicit placement."""
    return F.linear(F.silu(F.linear(inputs, tensors["gate_proj"])) *
                    F.linear(inputs, tensors["up_proj"]), tensors["down_proj"])


def reference_fp32(inputs, tensors):
    """Separate full-FP32 reference using explicit matmul and sigmoid SiLU."""
    x = inputs.float().cpu()
    gate = x @ tensors["gate_proj"].float().cpu().T
    up = x @ tensors["up_proj"].float().cpu().T
    return ((gate * torch.sigmoid(gate)) * up) @ tensors["down_proj"].float().cpu().T


def load_router(model_path, layer):
    """Read only the native gate tensor and its checkpoint routing contract."""
    root = Path(model_path).resolve()
    cfg = read_json_bounded(root / "config.json")
    index = read_json_bounded(root / "model.safetensors.index.json")
    if (cfg.get("model_type") != "qwen3_moe" or type(layer) is not int
            or not 0 <= layer < cfg["num_hidden_layers"] or layer in cfg.get("mlp_only_layers", [])
            or (layer + 1) % cfg.get("decoder_sparse_step", 1) != 0):
        raise ValueError("requested layer is not a Qwen3 sparse MoE layer")
    top_k = cfg["num_experts_per_tok"]
    if not 0 < top_k <= cfg["num_experts"] or type(cfg["norm_topk_prob"]) is not bool:
        raise ValueError("invalid native routing contract")
    name = f"model.layers.{layer}.mlp.gate.weight"
    relative = Path(index["weight_map"][name])
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or root not in path.parents or path.suffix != ".safetensors":
        raise ValueError("unsafe router shard path")
    shape = [cfg["num_experts"], cfg["hidden_size"]]
    if shape[0] * shape[1] * 2 > 8 * 1024**2:
        raise ValueError("router exceeds 8 MiB read limit")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        if view.get_shape() != shape or view.get_dtype() != "BF16":
            raise ValueError("router shape/dtype mismatch")
        weight = handle.get_tensor(name).clone().contiguous()
    return weight, cfg, {"tensor_name": name, "shard": str(path), "shape": shape,
                         "dtype": str(weight.dtype), "bytes": weight.numel() * weight.element_size(),
                         "sha256": tensor_hash(weight), "config_sha256": digest((root / "config.json").read_bytes()),
                         "index_sha256": digest((root / "model.safetensors.index.json").read_bytes())}


@dataclass(frozen=True)
class Routing:
    logits: object
    selected: object
    weights: object
    probabilities: object = None


def route(inputs, gate, top_k, normalize, *, capture_probabilities=False):
    """Qwen3's native gate order; selected IDs are never externally supplied."""
    if inputs.ndim != 2 or inputs.shape[1] != gate.shape[1] or inputs.dtype != gate.dtype:
        raise ValueError("router input shape/dtype mismatch")
    if (type(top_k) is not int or not 0 < top_k <= gate.shape[0] or type(normalize) is not bool
            or type(capture_probabilities) is not bool):
        raise ValueError("invalid routing parameters")
    logits = F.linear(inputs, gate)
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
    weights, selected = torch.topk(probabilities, top_k, dim=-1)
    if normalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return Routing(logits, selected, weights.to(inputs.dtype), probabilities if capture_probabilities else None)


def expert_groups(routing, *, mode='generic'):
    """Deterministic expert-ID order and native top-k-slot/token order."""
    if mode not in ('generic', 'single_token'):
        raise ValueError('unsupported expert grouping mode')
    if mode == 'single_token':
        if routing.selected.ndim != 2 or routing.selected.shape[0] != 1:
            raise ValueError('single-token grouping requires exactly one token')
        selected = routing.selected.detach().cpu()[0].tolist()
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('single-token grouping requires distinct selected expert IDs')
        tokens = torch.zeros((1,), dtype=torch.long, device=routing.selected.device)
        return [(expert, tokens, routing.weights[0,slot:slot+1])
                for slot,expert in sorted(enumerate(selected), key=lambda pair:pair[1])]
    groups = []
    # Native mask is [expert, top-k-slot, token]. Transpose preserves its order
    # without constructing an E*k*t one-hot tensor in the platform executor.
    for expert in sorted(set(routing.selected.detach().cpu().flatten().tolist())):
        slots, tokens = torch.where((routing.selected == expert).T)
        groups.append((expert, tokens, routing.weights[tokens, slots]))
    return groups


def collect_model_experts(model):
    """Keep immutable CPU tensor references before replacing native MoE blocks."""
    if model.config.model_type != "qwen3_moe":
        raise ValueError("only Qwen3 MoE supported")
    masters = {}
    for layer, decoder in enumerate(model.model.layers):
        if not hasattr(decoder.mlp, "experts"):
            raise ValueError("dense layers require a separate adapter")
        for expert, mlp in enumerate(decoder.mlp.experts):
            key = ExpertKey(layer, expert, "bf16")
            tensors = {name: getattr(mlp, name).weight.detach() for name in PROJECTIONS}
            if any(t.device.type != "cpu" or t.dtype != torch.bfloat16 for t in tensors.values()):
                raise ValueError("model masters must already be CPU BF16")
            masters[key] = ExpertWeights(key, tensors, {"source": "loaded native checkpoint CPU storage"})
    return masters


def pinned_master_layout(sizes, *, block_cap_bytes):
    """Whole-expert packing; the cap bounds requested tensors, not allocator rounding."""
    if (type(block_cap_bytes) is not int or not 0 < block_cap_bytes <= 1024**3
            or block_cap_bytes & (block_cap_bytes - 1) or not sizes
            or any(type(n) is not int or n <= 0 or n % 2 or n > block_cap_bytes for n in sizes.values())):
        raise ValueError('pinned master layout requires even expert bytes within a power-of-two block cap')
    blocks, current, used = [], [], 0
    for key in sorted(sizes, key=lambda k: (k.layer, k.expert, k.representation)):
        if used + sizes[key] > block_cap_bytes:
            blocks.append(tuple(current)); current, used = [], 0
        current.append(key); used += sizes[key]
    blocks.append(tuple(current))
    return tuple(blocks)


def pin_cpu_master_storage(masters, *, block_cap_bytes, check_budget):
    """Copy immutable CPU BF16 experts into bounded, owned pinned blocks.

    Views retain their backing block after this function returns. Allocation,
    registration and copying occur outside requests; the caller checks actual
    RSS, NUMA placement and full source/destination hashes. On failure the
    registry may be partially converted and must be discarded by the caller.
    """
    for key, weights in masters.items():
        if (weights.key != key or set(weights.tensors) != set(PROJECTIONS)
                or any(t.device.type != 'cpu' or t.dtype != torch.bfloat16 or not t.is_contiguous()
                       or t.is_pinned() for t in weights.tensors.values())):
            raise ValueError('pooled pinned copy requires unpinned contiguous CPU BF16 weights')
    sizes = {key: w.nbytes for key, w in masters.items()}
    groups = pinned_master_layout(sizes, block_cap_bytes=block_cap_bytes)
    records, copied_bytes, copied_experts = [], 0, 0
    for block_index, keys in enumerate(groups):
        payload_bytes = sum(sizes[key] for key in keys)
        check_budget()
        block = torch.empty(payload_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
        if not block.is_pinned() or block.device.type != 'cpu' or block.numel() != payload_bytes:
            raise RuntimeError('pinned block allocation did not meet the source contract')
        check_budget()
        offset = 0
        for key in keys:
            weights = masters[key]
            tensors, expert_start = {}, offset
            for name in PROJECTIONS:
                source = weights.tensors[name]
                nbytes = source.numel() * source.element_size()
                target = block[offset:offset + nbytes].view(source.dtype).view(source.shape)
                target.copy_(source)
                if (not target.is_pinned() or target.data_ptr() == source.data_ptr()
                        or target.data_ptr() != block.data_ptr() + offset or target.stride() != source.stride()):
                    raise AssertionError('pinned expert view aliases its source or changes packed layout')
                tensors[name] = target
                offset += nbytes
            masters[key] = ExpertWeights(key, tensors, {**weights.provenance,
                'storage': 'owned pooled pinned CPU BF16 copy; actual locality verified separately',
                'pinned_pool_block': block_index, 'pinned_pool_offset_bytes': expert_start})
            copied_bytes += sizes[key]; copied_experts += 1
            check_budget()
        if offset != payload_bytes:
            raise AssertionError('pinned expert packing does not conserve source bytes')
        records.append(dict(block=block_index, requested_bytes=payload_bytes,
            storage_bytes=block.untyped_storage().nbytes(), experts=len(keys),
            first_key=[keys[0].layer, keys[0].expert, keys[0].representation],
            last_key=[keys[-1].layer, keys[-1].expert, keys[-1].representation]))
    return dict(mode='pooled_pinned_copy', experts_copied=copied_experts, copied_bytes=copied_bytes,
                pinned_payload_bytes=copied_bytes, blocks=records, block_cap_bytes=block_cap_bytes,
                pool_tensor_storage_bytes=sum(r['storage_bytes'] for r in records),
                source_and_pinned_weights_are_one_copy=True,
                scope='owned pinned tensor bytes; allocator rounding/RSS/NUMA and original bytes require separate checks')


def copy_cpu_master_storage(masters, *, check_budget):
    """Own CPU BF16 bytes, replacing one expert at a time after module removal.

    The caller binds allocating threads and verifies actual NUMA placement.
    This function neither changes memory policy nor migrates shared file pages.
    """
    copied_bytes = 0
    for key in masters:
        weights = masters[key]
        if (weights.key != key or set(weights.tensors) != set(PROJECTIONS) or
                any(t.device.type != 'cpu' or t.dtype != torch.bfloat16 or not t.is_contiguous()
                    for t in weights.tensors.values())):
            raise ValueError('private master copy requires contiguous CPU BF16 weights')
        check_budget()
        tensors = {name: t.detach().clone(memory_format=torch.preserve_format) for name, t in weights.tensors.items()}
        if any(t.data_ptr() == weights.tensors[name].data_ptr() or
               t.stride() != weights.tensors[name].stride() for name, t in tensors.items()):
            raise AssertionError('private CPU copy aliases source or changes layout')
        masters[key] = ExpertWeights(key, tensors, {**weights.provenance,
            'storage': 'private CPU copy on caller-bound cores; placement verified separately'})
        copied_bytes += weights.nbytes
        check_budget()
    return {'experts_copied': len(masters), 'copied_bytes': copied_bytes,
            'source_storage_reused': False, 'strides_preserved': True,
            'shared_file_pages_modified': False}
