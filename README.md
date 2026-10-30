# PDScope

[English](README.md) | [简体中文](README_ZH.md)

Phase-adaptive cross-layer expert scheduling for memory-bound MoE inference on CPU–GPU
servers. Accepted at PPoPP 2027.

## Paper

The full paper is included in this repository:
[**PDScope: Breaking the Prefill-Decode Seesaw for Memory-Bound MoE Inference**](docs/pdscope-ppopp27.pdf)
(PPoPP 2027, Salt Lake City, March 20–24, 2027).

This is the arXiv version we circulated before the official ACM PPoPP proceedings were
printed. The proceedings version may differ in minor details, so cite the published paper
when the official version is available; see [Citation](#citation).

For artifact evaluators, [`docs/AE.md`](docs/AE.md) is a self-contained guide: hardware
requirements, a ~30-minute getting-started test, and a claim-by-claim map of what this
repository can and cannot reproduce.

## Overview

An MoE model activates only a few experts per token, but every expert weight still has to
be reachable. When the experts do not fit in GPU memory they live in host memory, and each
layer has to decide, expert by expert, whether to compute on the CPU or copy to the GPU.
Prefill and decode want opposite things — prefill wants as many tokens as possible computed
on the GPU, decode wants as few PCIe transfers as possible — so a single static scheduling
policy always loses on one of the two phases.

PDScope treats the phase as a first-class scheduling input and re-decides the operating
point when the phase changes. This repository is the reference implementation of the
system, built on the [ktransformers] CPU expert kernel:

- **AdaptSched — phase-adaptive cross-layer scheduling.** A cross-layer cost model ranks the
  demand of the current layer against the predicted demand of the next one and places
  experts under one shared PCIe/CPU budget; the decision is re-evaluated at the layer
  boundary as true routing replaces the prediction, and the prefetch depth follows the
  confidence of that prediction. Prefill runs an aggressive load/deep-pipeline operating
  point, decode a conservative one with a fixed CPU/GPU balance target.
- **Zero-overhead execution path.** The hot expert set is captured as resident CUDA graphs,
  expert copies go through zero-copy asynchronous I/O with demand staging slots, and the
  decode path hands a whole layer's CPU expert group to one native C++ call instead of ~20
  Python/ATen operations. Fused merge reduces the per-expert scatter to one weighted
  index-add.
- **Predictor.** This repository ships the causal gate-fallback predictor (the next layer's
  gate applied to the current activation). The paper's learned phase-aware predictor is not
  included; every trace row records that substitution.

## Key Results

`Qwen3-30B-A3B` (BF16, 48 layers, 128 experts, top-8) on one A100 80 GB PCIe with the
process capped at 24 GiB of GPU memory. Single request, 64 generated tokens, frozen
ShareGPT inputs taken from the paper's latency workload:

| input tokens | prefill tok/s | decode tok/s | end-to-end tok/s | GPU peak |
|---:|---:|---:|---:|---:|
| 512  | 343.92 | 7.984 | 6.821 | 19.98 GiB |
| 1024 | 597.18 | 7.689 | 6.460 | 20.12 GiB |
| 2048 | 1086.83 | 7.902 | 6.493 | 20.37 GiB |
| 4096 | 1671.54 | 7.765 | 6.058 | 20.83 GiB |

Best single-request decode observed with this configuration: **8.04 tok/s**. These are
single-request numbers from one host; the paper additionally reports the multi-model,
multi-batch comparison against other systems, which this repository does not reproduce.

## Quick Start

### 1. Requirements

| item | reference machine |
|---|---|
| GPU | 1 × NVIDIA A100 80 GB PCIe; the process is capped at 24 GiB |
| CPU | 2 sockets × 24 physical cores (Intel Xeon Gold 5318Y, 2 NUMA nodes) |
| RAM | ≥ 256 GiB — the pinned BF16 expert masters alone are ~58 GiB |
| OS / libc | any current x86-64 Linux (measured on RHEL 9, glibc 2.34) |
| Python | 3.11, with `torch 2.10.0+cu128` and `transformers 4.51.3` (see `requirements.txt`) |
| Compiler | g++ 11+ and Python headers; no `nvcc` needed |
| Model | `Qwen3-30B-A3B` BF16 safetensors, downloaded locally |
| Dataset | `ShareGPT_V3_unfiltered_cleaned_split.json` |

A different GPU with ≥ 24 GiB or a different core count works, but the decode CPU/GPU split
and the worker count are properties of the host and have to be re-tuned.

### 2. Install

```bash
git clone https://github.com/nezha-Anonymous/PDScope.git pdscope
cd pdscope
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

### 3. Build the CPU expert kernel and extensions

```bash
git clone https://github.com/kvcache-ai/ktransformers.git "$HOME/kt-src"
git -C "$HOME/kt-src" checkout a02efe36b5802e7ca79a6b29146e01cf3b97d305
git -C "$HOME/kt-src" submodule update --init --recursive
export KT_SOURCE_ROOT="$HOME/kt-src"

export PYTHON_BIN="$(command -v python)"
./scripts/build_native.sh
```

Everything is compiled against your own toolchain into `expertbench/native-build/`, so no
system upgrade is involved. `WITH_AVX512=1 ./scripts/build_native.sh` also builds an AVX-512
variant; the AVX2 build stays the default because the kernel streams ~66 MB per decode layer
and is DRAM-bandwidth bound.

### 4. Point at your model and dataset

```bash
export PDSCOPE_MODEL=/path/to/Qwen3-30B-A3B
export PDSCOPE_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 5. Run

```bash
./scripts/run_reference.sh               # -> runs/<UTC timestamp>/result.json
./scripts/run_reference.sh results/run1  # -> results/run1/result.json
```

The four lengths take about five minutes in total, model loading included. Each run writes
`result.json` (throughput, memory, full configuration), `manifest.json` (host, package
versions, source snapshot), `prompt_sources.json` (the frozen requests) and
`traces/<case>.json.gz` (per layer: placement decision, prefetch ledger, host timings).

The equivalent direct invocation, useful when you want to change `--lengths`, is:

```bash
python benchmarks/pdscope_compare.py \
  --lengths 512 1024 2048 4096 --repetitions 2 \
  --pdscope-cpu-threads 18 --pdscope-decode-cpu-targets 6
```

The measured cost table, the hot-expert table and the built kernel are picked up
automatically, the execution path defaults to the measured reference configuration,
and `--output` defaults to `runs/<timestamp>/result.json`.

### 6. Options worth knowing

| flag | effect |
|---|---|
| `--pdscope-cpu-threads N` | worker threads per NUMA subpool (default 18, i.e. 36 total). **The most important knob**: with all physical cores handed to the kernel the control thread lands on SMT siblings, every host call inflates ~4×, and decode drops ~20 %. |
| `--pdscope-decode-cpu-targets T` | decode keeps `T` of the 8 active experts on the CPU (6 on the reference host). Pass several values to try them in one session. Sweep it on your machine. |
| `--cache-slots S` | experts kept resident per layer (default 40 = 1920 slots, 16.9 GiB). The GPU cap is the real limit. |
| `--speculative-slots S` | how many prefetched experts may occupy the cache beyond the static hot set (default 128). |
| `--pdscope-demand-slots S` | demand staging depth, i.e. expert copies in flight (default 4). |
| `--attn-implementation` | attention backend (default `sdpa`: it frees the eager softmax workspace, which is what lets 40 slots fit at 4096-token inputs). |
| `--no-pdscope-graphs`, `--no-pdscope-native-dispatch`, `--no-pdscope-input-pregather`, `--no-pdscope-graph-output-pool`, `--no-torch-expandable-segments` | switch a default optimization off, e.g. for an ablation against your own system. |

`--help` lists everything else.

### 7. Check a build without loading the whole model (optional)

```bash
python benchmarks/pdscope_compare.py --mode two_layer
```

Real layers 0/1 with a synthetic activation: routing against an independent one-hot dispatch,
native-vs-FP32 relative L2 ≤ 0.02, DMA byte closure, slot lifetimes and cache-handle release.

## Notes and limitations

- **Predictor.** The causal gate-fallback predictor replaces the paper's learned phase-aware
  predictor; every trace row records `predictor_prior_hit_rate=0`.
- **Scope.** Single request, one model (`Qwen3-30B-A3B`), one GPU. Multi-batch serving and
  the other models from the paper are out of scope here.
- **Host sensitivity.** Worker count and unrelated load move throughput by tens of percent.
  Measure on an idle machine and re-tune before comparing with the table above; the
  measurement protocol is described in [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Citation

PDScope appears at the ACM SIGPLAN Symposium on Principles and Practice of Parallel
Programming (PPoPP) 2027, Salt Lake City, March 20–24, 2027.

```bibtex
@inproceedings{pdscope2027,
  title     = {PDScope: Breaking the Prefill-Decode Seesaw for Memory-Bound MoE Inference},
  author    = {Yu, Enda and Dong, Dezun and Zhang, Zhaoning and Yang, Xiaojian and Yang, Weiling and Ma, Shaonan and Li, Dongsheng and Wu, Yongwei and Liao, Xiangke},
  booktitle = {Proceedings of the ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming (PPoPP '27)},
  year      = {2027}
}
```

## License

Apache-2.0 (see `LICENSE`). The ktransformers CPU expert kernel is fetched at a pinned commit
and built locally; it is not redistributed here.

[ktransformers]: https://github.com/kvcache-ai/ktransformers
