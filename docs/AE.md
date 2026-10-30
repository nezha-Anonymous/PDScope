# Artifact evaluation guide (PPoPP 2027)

This document is addressed to the PPoPP 2027 artifact-evaluation committee. It says
what is in the artifact, how to get it running in about 30 minutes, how to reproduce the
numbers this artifact reports, and which claims of the paper the artifact can and cannot
support.

- **Paper**: *PDScope: Breaking the Prefill-Decode Seesaw for Memory-Bound MoE Inference*
  (PPoPP 2027). The version supplied with this submission is
  [`docs/pdscope-ppopp27.pdf`](pdscope-ppopp27.pdf) — the arXiv version circulated before
  the official ACM proceedings were printed; the proceedings version may differ in minor
  details.
- **Artifact**: this repository, Apache-2.0. It is the reference implementation of the
  PDScope execution engine and scheduler, built on the pinned ktransformers CPU expert
  kernel. Public mirror: <https://github.com/nezha-Anonymous/PDScope>.
- **Scope in one line**: the artifact runs PDScope itself on `Qwen3-30B-A3B` for
  single-request prefill/decode and exposes per-layer decisions and host timings; it does
  **not** ship the baseline systems, the other three models, the multi-batch path, or the
  learned PhasePer predictor.

---

## 1. What is in the artifact

| Path | Content |
|---|---|
| `benchmarks/pdscope_compare.py` | entry point: measurement protocol (`--mode measure`) and a numerical self-check (`--mode two_layer`) |
| `expertbench/pdscope_policy.py` | AdaptSched (paper Algorithm 1) as a pure function: cross-layer cost model, layer-local re-evaluation, confidence-aware prefetch depth |
| `expertbench/pdscope_runtime.py` | per-layer execution: demand declaration, placement, prefetch ledger, CUDA-graph hot set, merge |
| `expertbench/executor.py` | pinned master copies, staging slots, asynchronous DMA, resident CUDA graphs |
| `expertbench/cache.py` | expert cache ledger, static hot set, demand accounting |
| `expertbench/adapters/qwen.py` | Qwen3 MoE adapter (routing, expert weights, BF16 expert math) |
| `baselines/ktransformers/runtime.py` | CPU backend adapter around the pinned ktransformers kernel, incl. the native decode dispatch |
| `native/` | the native layer-dispatch extension (ATen + CUDA runtime) and the build recipes for the CPU kernel |
| `costs/cost_table.json` | measured per-expert cost table (8 load points) that the scheduler's cost model interpolates |
| `hot/qwen.txt` | static hot-expert table from an offline profile |
| `results/reference_measurements.{md,json}` | the numbers this artifact reports, and their configuration |
| `docs/REPRODUCE.md` | measurement protocol, host sensitivity, tuning order |

Nothing is prebuilt: no binaries and no model weights are shipped.

## 2. Requirements

| Resource | Requirement | Measured on the reference host |
|---|---|---|
| GPU | any NVIDIA GPU with **≥ 24 GiB** for the reference configuration (a 16 GiB card runs a reduced configuration, see §7). PCIe 4.0 x16 measured. | 1 × A100 80 GB PCIe, process capped at 24 GiB |
| Host RAM | **≥ 128 GiB**, because the BF16 expert masters are pinned in host memory by design (~58 GiB) | peak RSS **112 GiB** |
| CPU | ≥ 2 physical cores; the decode/prefill optimum moves with core count | 2 sockets × 24 cores (Xeon Gold 5318Y), 36 worker threads |
| Disk | ~70 GB free (model ~60 GB + dataset ~0.6 GB) | — |
| OS / libc | current x86-64 Linux; **no nvcc** needed (the extension uses ATen + the CUDA runtime) | RHEL 9, glibc 2.34 |
| Software | Python 3.11, `torch 2.10.0+cu128`, `transformers 4.51.3` (see `requirements.txt`), `g++` ≥ 11 with Python headers | — |
| Data | `Qwen3-30B-A3B` BF16 safetensors, and `ShareGPT_V3_unfiltered_cleaned_split.json` | — |

The paper's evaluation ran on a dual Xeon Gold **6230R** + A100 and on an RTX 3090 +
EPYC 7452 (PCIe 4.0 x16). This artifact is measured on a dual **5318Y** + A100. The
scheduling mechanism and the direction of every effect carry over, absolute throughput
does not — see §6.

Model weights and the ShareGPT file are not redistributable here. Both are public:
`Qwen/Qwen3-30B-A3B` (HF) and the `ShareGPT_V3_unfiltered_cleaned_split.json` used by
several MoE inference papers.

## 3. Getting started (about 30 minutes)

```bash
git clone https://github.com/nezha-Anonymous/PDScope.git pdscope
cd pdscope
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt                        # ~2-5 min
```

**Build the two native pieces** (~10–20 min, CPU only):

```bash
git clone https://github.com/kvcache-ai/ktransformers.git "$HOME/kt-src"
git -C "$HOME/kt-src" checkout a02efe36b5802e7ca79a6b29146e01cf3b97d305
git -C "$HOME/kt-src" submodule update --init --recursive
export KT_SOURCE_ROOT="$HOME/kt-src"
export PYTHON_BIN="$(command -v python)"
./scripts/build_native.sh        # -> expertbench/native-build/{kt-kernel,layer-dispatch}/*.so
```

**Point the tool at your data** (no download is provided by the artifact):

```bash
export PDSCOPE_MODEL=/path/to/Qwen3-30B-A3B
export PDSCOPE_DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
```

**Smoke test** — the numerical self-check; real layers 0/1, a synthetic activation, no
full model, **about 4 minutes**:

```bash
python benchmarks/pdscope_compare.py --mode two_layer
```

Expected: `{"status": "completed", ...}` and `rows` with 10 entries. Per row the check
asserts routing equality against an independent one-hot dispatch, native-vs-FP32 relative
L2 ≤ 0.02, DMA byte closure, slot lifetimes and cache-handle release. On the reference
host the maximum `native_queue_vs_fp32.relative_l2` is **0.005–0.006**, and
`cache_stats.h2d_bytes_submitted == cache_stats.h2d_bytes_completed`. A failing build, a
missing kernel hook or a broken stream schedule fails this check loudly.

**First real measurement** (optional in the 30-minute budget; see §4) — one input length,
**about 4 minutes** after the model is on the page cache:

```bash
python benchmarks/pdscope_compare.py --lengths 512 --repetitions 1
```

## 4. Reproducing the artifact's numbers

The full reference run (four input lengths, two warm-up plus two measured requests each,
64 generated tokens) is:

```bash
./scripts/run_reference.sh               # -> runs/<UTC timestamp>/result.json
```

Expected wall clock: **5–12 min** once the model is in the page cache, plus a one-time
2–25 min for loading and pinning the expert masters (depends on page-cache warmth; on the
reference host, warm 2–3 min, cold up to 10 min).

Expected results on the reference host (otherwise idle machine):

| input tokens | prefill tok/s | decode tok/s | end-to-end tok/s | GPU peak |
|---:|---:|---:|---:|---:|
| 512  | 394.89 | 8.069 | 6.9 | 19.99 GiB |
| 1024 | 721.49 | 7.976 | 6.6 | 20.13 GiB |
| 2048 | 1238.33 | 7.876 | 6.5 | 20.38 GiB |
| 4096 | 1937.93 | 7.852 | 6.1 | 20.84 GiB |

Acceptable variation, and where it comes from:

- **Same host, same day**: ±4 % decode, ±5–10 % prefill (two measured requests per point).
- **Different host**: prefill and decode both move by tens of percent. The two knobs that
  matter are `--pdscope-cpu-threads` (worker threads per NUMA subpool; more workers than
  physical cores pushes the control thread onto SMT siblings and costs ~20 % decode) and
  `--pdscope-decode-cpu-targets` (decode CPU/GPU balance). `docs/REPRODUCE.md` §4 gives the
  sweep order; `--pdscope-decode-cpu-targets 4 5 6 7 8` tries several targets in one
  session.
- **Unrelated load on the host**: prefill moves first. Measure on an idle machine; the
  artifact records the host load context in `manifest.json`.

How to read the output: `result.json → requests[i] → metrics` holds
`prefill_tokens_per_second`, `decode_tokens_per_second`,
`end_to_end_output_tokens_per_second`, the per-token timestamps in `token_ready_seconds`,
and `gpu_peak_reserved_bytes`. `traces/<case>.json.gz` holds one row per layer (see §5).
The first generated token is counted as prefill, so decode is measured over the following
63 forwards.

## 5. Claim-by-claim map

| Paper item | Reproducible here | How to check |
|---|---|---|
| Fig. 1, 9, 10, 12 — throughput vs KTransformers, Fate, Fiddler, HybriMoE, MoE-APEX | **No** | The baseline systems are not shipped with this artifact and are not redistributable by us; the artifact contains PDScope only. This is the largest gap for a *Results Reproduced* claim. |
| Fig. 11 — DeepSeek-V4-Flash, multi-batch scaling | **No** | Model not shipped; the released harness is single-request by design. |
| Fig. 13, Table 5 — PhasePer prediction accuracy | **No** | The learned phase-aware predictor is not part of this artifact; it is replaced by a causal gate-fallback predictor (next layer's gate on the current activation). Every trace row records `predictor_prior_hit_rate=0` so the substitution is visible. |
| Fig. 14 — runtime breakdown | **Partly** | The PDScope-side breakdown is reproducible from traces (`counters`, `balance.host_ms`, per-layer wall time); the KTransformers/HybriMoE columns are not. |
| §4.3, Fig. 7–8 — phase-adaptive decisions (aggressive prefill, conservative decode) | **Yes** | Run with `--pdscope-balance`. `result.json → requests[i] → phase_balance` aggregates per phase: `work.cpu_experts` / `work.gpu_experts`, token assignments, H2D bytes, prefetch loads. On the reference host a 512-token request shows prefill issuing a large on-demand GPU set per layer (`decision.mode = "prefill"`, e.g. 33 CPU / 41 resident / 23 on-demand experts at layer 5) while decode runs `c_unload` with a small CPU set and **zero** H2D bytes for most layers. The same fields are per layer in `traces/*.json.gz → decision`. |
| §4.4 — zero-overhead execution path (resident CUDA graphs, AsyncIO, native decode dispatch) | **Yes, as A/B deltas** | The default path is on; each piece can be switched off: `--no-pdscope-graphs`, `--no-pdscope-native-dispatch`, `--no-pdscope-input-pregather`, `--no-pdscope-graph-output-pool`. Measure each condition with `--pdscope-balance` and compare `balance.host_ms` (`cpu_finish_wait_ms`, `merge_submit_ms`, `resident_gpu_submit_ms`) and end-to-end tok/s at fixed input length. Effect sizes depend on the host; on the reference host the decode path's host-side windows, not the expert math, are what these switches move. |
| Table 1 — PCIe single-expert transfer latency | **No** (not instrumented as a microbenchmark) | H2D bytes per layer are in the traces, so an effective bandwidth can be derived, but there is no dedicated PCIe microbenchmark in the artifact. |
| Table 3 — AsyncIO bandwidth utilization | **Partly, derived** | `counters.h2d_bytes_submitted` per layer divided by the corresponding host window gives an effective rate; same caveat as above. |
| Numerical correctness of the fused merge and the native path | **Yes** | `--mode two_layer` (§3). |
| Absolute single-request throughput for Qwen3-30B-A3B | **Yes** | §4. |

**Supported:** the artifact installs from a clean checkout, builds with the host toolchain,
runs PDScope on Qwen3-30B-A3B, reproduces the absolute single-request prefill/decode
numbers of §4 within the quoted variance, exposes the per-layer placement/prefetch
decisions and host timings that the phase-adaptive claims rest on, and passes the
numerical check in `--mode two_layer`.

**Not supported by this artifact:** any speedup *relative to another system*, the other
three models, multi-batch serving, and the learned predictor's accuracy. We do not claim
the paper's headline ratios can be reproduced from this repository alone.

## 6. Interpreting differences from the published numbers

The paper measures on a dual Xeon Gold 6230R + A100 and an RTX 3090 + EPYC 7452; this
artifact is measured on a dual Xeon Gold 5318Y + A100. Both phases are sensitive to the
host in different ways, so expect:

- **prefill** to scale with free physical cores and with the CPU's effective DRAM bandwidth;
- **decode** to be bounded by the per-layer host sequence plus PCIe copy latency, and to
  drop sharply if the control thread shares cores with worker threads;
- both to drift under unrelated host load. If your prefill is far below §4, check
  `--pdscope-cpu-threads` first (try 8, 12, 16, 18, 20, 24 per subpool), then
  `--pdscope-decode-cpu-targets`, then whether the machine is otherwise idle.

The artifact reports what it measured, including the GPU peak and the host RSS, so a
mismatch is diagnosable rather than silent.

## 7. Running on smaller hardware

The host-RAM requirement is not negotiable (the expert masters are pinned by design), but
the GPU configuration is:

```bash
# 16 GiB card, two lengths, one measured request per length
python benchmarks/pdscope_compare.py --lengths 512 1024 --repetitions 1 \
  --cache-slots 16 --gpu-total-bytes $((16*1024**3)) --pdscope-cpu-threads 8
```

`--cache-slots S` reserves `S × 48 × 9 MiB` on the GPU (40 slots = 16.9 GiB; 16 slots =
6.75 GiB; 8 slots = 3.4 GiB) on top of ~3.2 GiB of non-expert state. Lower values reduce
the resident hot set, so decode throughput falls accordingly — the mechanism still runs,
the absolute numbers change. 4096-token inputs additionally need the eager softmax
workspace to fit, which is why the reference configuration uses SDPA attention.

## 8. Adapting the artifact (for a Reusable claim)

- **Different input lengths / repetitions**: `--lengths`, `--repetitions`; the frozen
  input protocol (paper's `latency.py` style) is described in `docs/REPRODUCE.md` §1 and
  recorded per request in `prompt_sources.json`.
- **Different hardware**: re-tune `--pdscope-cpu-threads`, `--pdscope-decode-cpu-targets`,
  `--cache-slots`, and re-measure `costs/cost_table.json` if the CPU kernel or the core
  count changes — the cost model is a property of the host
  (`docs/REPRODUCE.md` §4).
- **Different scheduling policy**: `expertbench/pdscope_policy.py` is a pure function
  (`plan_pdscope`) with no clocks and no tensors; the runtime calls it once per layer, so a
  new policy can be dropped in and A/B-tested through the same harness.
- **Different model**: the adapter is Qwen3-specific (`expertbench/adapters/qwen.py`,
  48 layers × 128 experts × top-8 is asserted at load time); supporting another MoE model
  means a new adapter and a new hot table.
- **Your own system comparison**: the harness measures one configuration per process and
  writes a complete, self-describing `result.json` + `manifest.json`; the `--no-*` switches
  above give controlled ablations of each optimization.

## 9. Provenance and contacts

- Artifact commit used for the reported numbers: see `manifest.json` of any run (it records
  `git_commit`, `git_status`, the source snapshot hash and the native build hashes).
- License: Apache-2.0 (`LICENSE`). The ktransformers CPU kernel is fetched at the pinned
  commit `a02efe3` and built locally; it is not redistributed here.
- Questions during evaluation: use the HotCRP discussion for this submission. Contact
  addresses of all authors are listed in `CITATION.cff`.
