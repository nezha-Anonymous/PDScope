# Measurement protocol, caveats and tuning notes

## 1. Workload

One request at a time, frozen inputs, no dataset averaging:

- Inputs follow the protocol of the paper's `latency.py`: filter non-empty conversations,
  take the first user turn, normalise whitespace, shuffle with `random.Random(0)`, then
  for each requested length take the first text with at least that many words and
  tokenize it with the model's own tokenizer, truncated to the length. No chat template
  is added and no padding is applied. `prompt_sources.json` records, per request, the
  dataset row, the original conversation id, the shuffle position, the text and token
  hashes and the token ids, so a run can be replayed exactly.
- `--output-tokens 64` with greedy decoding. The first generated token belongs to
  prefill, so decode throughput counts the following 63 forwards.
- Two full warm-ups of the same configuration and input precede `--repetitions`
  measured requests. With two measures the run-to-run spread is ±4 % for decode and
  ±5–10 % for prefill; raise `--repetitions` if you need tighter error bars.

## 2. Memory and resource accounting

- The process is capped at 24 GiB of GPU memory (`torch.cuda.set_per_process_memory_fraction`).
  Each case reports `gpu_peak_reserved_bytes`, the cache pool reservation, the demand
  staging reservation and the process-external bytes, and fails if their sum exceeds the
  cap.
- At 40 slots per layer the expert pool is 16.9 GiB and the observed peak is
  19.98–20.83 GiB depending on input length, because the attention workspace grows with
  context.
- 4096-token inputs need `--torch-expandable-segments`. Without it the same configuration
  fails on ~3.96 GiB of *reserved-but-unallocated* fragmentation rather than on a real
  shortfall.

## 3. Host sensitivity

The control thread runs the whole per-layer host sequence, so anything that occupies the
physical cores slows it down.

- **CPU worker count dominates.** With 2 × 24 = 48 workers (every physical core) the
  control thread ends up on SMT siblings and each host call inflates ~4×; that measured
  6.17 decode and 453 prefill tok/s at 1024 tokens. With 2 × 18 = 36 workers the same
  configuration measured 7.75 decode and 609 prefill. Sweep it on your machine.
- **Unrelated load matters.** A load average of 15–19 from other work moved 1024-token
  prefill from ~610 to ~560 tok/s across sessions at a fixed configuration. Measure on an
  otherwise idle host.
- **Both phases are tuned separately, on purpose.** The deep demand pipeline is
  prefill-only; the native decode dispatch and the decode CPU/GPU target are decode-only.

## 4. Tuning order on a new machine

1. `--pdscope-cpu-threads N`: sweep 8, 12, 16, 18, 20, 24 threads per subpool and pick
   the point where the expert kernel still saturates DRAM but several physical cores stay
   free. Watch `cpu_native_span_ms` (kernel time) and the decode host windows
   (`--pdscope-balance`).
2. `--pdscope-decode-cpu-targets T`: with the worker count fixed, sweep 4…8. On the
   reference host 6 was best at 36 workers and 7 at 48.
3. `--cache-slots S`: raise until the reported peak approaches the cap; long inputs need
   a lower value. The pool is one contiguous allocation, so 44 slots is the practical
   maximum at ≤2048 tokens and 48 fails outright.
4. `costs/cost_table.json`: the measured per-expert cost table shipped with this
   repository, calibrated at 36 workers. Re-measure it if you change the CPU kernel or the
   core count — the scheduler's cross-layer cost model is a property of the host. The
   explicit `--pdscope-decode-cpu-targets` override still works with a stale table.

`--pdscope-decode-cpu-targets` accepts several values and runs them back to back inside
one process (same scheduler state, same inputs), which is the cheapest way to settle a
tuning question without cross-session noise.

## 5. Verification hooks

- `--mode two_layer` runs real layers 0/1 with a synthetic activation and checks, per row:
  routing equality against an independent one-hot dispatch, native-vs-FP32 relative
  L2 ≤ 0.02, DMA byte closure, slot lifetimes and cache-handle release. With
  `--pdscope-merge fused` the row must instead satisfy the explicit accumulation-order
  bound (`relative_l2 ≤ 1e-2`), because the fused sum is not bitwise.
- Every measured case fails if a requested host path did not reach the runtime, so a flag
  that silently defaults cannot be mistaken for an A/B result.
- Traces record, per layer: the placement decision, the prefetch ledger, the prediction
  hit counters, the cache counters and the host timing windows.
