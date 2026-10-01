# H2H Multi-Host Benchmark

Cross-host H2H on the BAP multi-host runner (`linux-x86-ct5lp-4tpu-x4`): worker 0
sends KV blocks over the NIC to worker 1 with the C++ `h2h_benchmark_runner`;
workers 2-3 idle. Driver: `//tpu_sync/benchmarks:h2h_multihost_bench`.

## 1. Analyze — measure the distribution, pick the configs

Dispatch `.github/workflows/h2h_multihost_analyze.yml`
(registry: `benchmark_registry_h2h_multihost_analyze.pbtxt`).

Output (BAP artifact `artifacts-<job_id>`, also on the `benchmark-data` branch):
`h2h_multihost_report.md` has one row per config. Keep the configs marked
`suitable`; `p_below_floor` is the estimated false-alarm rate of the gate.

The end of `h2h_multihost_report.md` prints the lines for the next step:
a `--configs=bs:nb:p,...` flag and one `metrics { ... }` line per suitable
config. Paste them into **both** `benchmark_registry_h2h_multihost_record.pbtxt`
and `benchmark_registry_h2h_multihost_gate.pbtxt`: `--configs=...` replaces
`--suite=...` in `runtime_flags`, and the `metrics` lines replace the existing
ones. Commit, then continue.

## 2. Record — compute the thresholds

Dispatch `.github/workflows/h2h_multihost_record.yml`
(registry: `benchmark_registry_h2h_multihost_record.pbtxt`, now on the chosen configs).

Copy `h2h_multihost_baselines.json` from the artifact to
`tpu_sync/benchmarks/h2h_multihost_baselines.json` and commit it.
`floor_gbs = max(median - 3.5 * MAD_sigma, 0.95 * median)` per config.

## 3. Gate

`.github/workflows/h2h_multihost_gate.yml` runs nightly (02:00 UTC) and on
dispatch (registry: `benchmark_registry_h2h_multihost_gate.pbtxt`, same configs as record).

Fails on `DATA CORRUPTION` (receiver byte check), `NO MEASUREMENT`,
`NO VERDICT`, or `BELOW FLOOR`. A config without a floor prints `NO FLOOR`
and passes.
