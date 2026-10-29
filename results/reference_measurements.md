# Reference measurements (single request, 64 output tokens)

Measured 2026-10-28 on an otherwise idle host with the configuration below; the
per-request timings and traces of that session are the ones summarised here.

| input tokens | prefill tok/s | decode tok/s | end-to-end tok/s | prefill s | decode s | GPU peak |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 343.92 | 7.984 | 6.821 | 1.493 | 7.891 | 19.98 GiB |
| 1024 | 597.18 | 7.689 | 6.460 | 1.715 | 8.193 | 20.12 GiB |
| 2048 | 1086.83 | 7.902 | 6.493 | 1.884 | 7.973 | 20.37 GiB |
| 4096 | 1671.54 | 7.765 | 6.058 | 2.450 | 8.113 | 20.83 GiB |

Configuration: see `results/reference_measurements.json`. Protocol and tuning notes:
`docs/REPRODUCE.md`. Reproduce with `./scripts/run_reference.sh`.
