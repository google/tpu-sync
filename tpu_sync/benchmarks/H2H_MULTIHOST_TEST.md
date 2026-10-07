# H2H Multi-Host Benchmark

Cross-host H2H gate on the multi-host TPU runner (`linux-x86-ct5lp-4tpu-x4`):
worker 0 sends KV blocks over `eth0` to worker 1 via `h2h_benchmark_runner`;
workers 2–3 exit immediately. Driver: `//tpu_sync/benchmarks:h2h_multihost_bench`.

## Gate Coverage & Where It Runs

Runs the 3 `--suite=correctness` configs (`{block_size}B_x{num_blocks}_P{parallelism}`:
`1048576B_x64_P1`, `1048576B_x64_P8`, `1048573B_x64_P4`) in:
- Postsubmit (`.github/workflows/postsubmit_benchmarks.yml`)
- Nightly (`.github/workflows/nightly_benchmarks.yml`, `00:00` UTC daily)
- Manual dispatch (`.github/workflows/run_benchmarks.yml`)

## Throughput Floors & Re-Recording

`floor_gbs = max(median - 3.5 * MAD_sigma, 0.90 * median)` (`--max_margin=0.10`).
Only `1048576B_x64_P1` (`P=1`) carries a throughput floor in
`h2h_multihost_baselines.json`; `P > 1` configs (`1048576B_x64_P8`,
`1048573B_x64_P4`) have 18–27% cross-run spread from multi-stream TCP flow
placement across pod virtual NIC queues and run with `NO FLOOR` (byte-integrity).

To re-record floors: dispatch `.github/workflows/h2h_multihost_record.yml`,
keep only `1048576B_x64_P1` in `h2h_multihost_baselines.json`, and commit it.

## Gate Verdicts

Fails on `DATA CORRUPTION` (receiver byte mismatch), `NO MEASUREMENT`,
`NO VERDICT`, or `BELOW FLOOR`. Unlisted configs run `NO FLOOR` (integrity only).

