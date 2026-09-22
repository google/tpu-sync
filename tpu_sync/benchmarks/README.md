# TPU Sync Benchmarking Architecture & Onboarding Guide

This document outlines the architecture and maintenance process for TPU Sync's
(formerly TPU Raiden) open-source microbenchmarking suite. We leverage the
[Benchmarking Automation Platform (BAP)](https://github.com/google-ml-infra/bap)
via GitHub Actions to execute workloads across AI accelerators available as
GitHub Actions runners and publish metrics for performance regression tracking
and presubmit gating.

---

## Quick Links

* [TPU Sync Benchmark Registry (`benchmark_registry.pbtxt`)](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/benchmark_registry.pbtxt)
* [BAP Metrics Module (`bap_metrics.py`)](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/bap_metrics.py)
* [Presubmit Workflow (`presubmit_benchmarks.yml`)](https://github.com/google/tpu-sync/blob/main/.github/workflows/presubmit_benchmarks.yml)
* [Postsubmit Workflow (`postsubmit_benchmarks.yml`)](https://github.com/google/tpu-sync/blob/main/.github/workflows/postsubmit_benchmarks.yml)
* [Nightly Workflow (`nightly_benchmarks.yml`)](https://github.com/google/tpu-sync/blob/main/.github/workflows/nightly_benchmarks.yml)
* [Run Benchmarks Manual Workflow (`run_benchmarks.yml`)](https://github.com/google/tpu-sync/blob/main/.github/workflows/run_benchmarks.yml)
* [MLCompass Dashboard](http://go/tpu-sync-oss-mlcompass)
* [BAP (Benchmarking Automation Platform) Repository](https://github.com/google-ml-infra/bap)
* [BAP General Onboarding Guide](https://github.com/google-ml-infra/bap/blob/main/docs/onboarding.md)
* [BAP Registry Definition (`benchmark_registry.proto`)](https://github.com/google-ml-infra/bap/blob/main/bap_proto/benchmark_registry.proto)

---

## Overview

The TPU Sync microbenchmarking suite is designed to automatically detect
performance regressions on pull requests (`presubmit` gating) and track
performance trends across continuous post-merge (`postsubmit`) runs on physical
hardware.

Under BAP, all benchmark configurations, hardware variants, execution action
paths, and metric definitions are centralized in a single declarative Protocol
Buffer text file:
[`tpu_sync/benchmarks/benchmark_registry.pbtxt`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/benchmark_registry.pbtxt).

---

## System Architecture

Our benchmarking pipeline relies on a push-based execution and ingestion model
consisting of four main components:

1. **Benchmark Registry**
   ([`benchmark_registry.pbtxt`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/benchmark_registry.pbtxt)):
   The declarative source of truth for all workloads, target metrics, stat
   definitions (e.g., `MEAN`), and environment configurations.
2. **Workload Execution & Metrics Emission**:
   * Workloads are executed via the [Bazel executor](https://github.com/google-ml-infra/bap/blob/main/docs/onboarding.md#bazel-executor).
   * Telemetry is emitted via [`bap_metrics.py`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/bap_metrics.py). It writes V1
     TensorBoard scalar events (`events.out.tfevents.*`) directly to
     `$TENSORBOARD_OUTPUT_DIR` using `TFRecordWriter`. This avoids Python GIL
     interference during timing loops and ensures reliable parsing by BAP's
     `tb_parser`.
3. **GitHub Actions Workflows**: The continuous integration pipelines
   ([`presubmit_benchmarks.yml`](https://github.com/google/tpu-sync/blob/main/.github/workflows/presubmit_benchmarks.yml),
   [`postsubmit_benchmarks.yml`](https://github.com/google/tpu-sync/blob/main/.github/workflows/postsubmit_benchmarks.yml),
   [`nightly_benchmarks.yml`](https://github.com/google/tpu-sync/blob/main/.github/workflows/nightly_benchmarks.yml), and
   [`run_benchmarks.yml`](https://github.com/google/tpu-sync/blob/main/.github/workflows/run_benchmarks.yml))
   invoke BAP's core engine, package execution logs and benchmark results
   (`results.json`), and generate markdown summary reports in the workflow runs.
4. **Pub/Sub Metric Publishing**: BAP publishes structured JSON result payloads
   to a Google Cloud Pub/Sub topic for downstream ingestion and tracking.

---

## Running Benchmarks

### Manual / Ad-hoc Runs via GitHub Actions (`run_benchmarks.yml`)

If you need to execute benchmarks on demand—such as testing a registry change on
a feature branch, verifying performance fixes, or running A/B comparisons—you can
trigger manual workflows via `workflow_dispatch` in the GitHub Actions UI:

1. **Push to a remote branch**: Commit your changes and push your feature branch
   to the repository.
2. **Trigger the workflow**: Navigate to the manual workflow (**Run Benchmarks** /
   [`run_benchmarks.yml`](https://github.com/google/tpu-sync/blob/main/.github/workflows/run_benchmarks.yml))
   in GitHub Actions and click **Run workflow**.
3. **Configure Options**:
   * **`ab_mode`**: Enable to run an A/B comparison (baseline vs experiment).
   * **`baseline_ref`**: Git ref for baseline (defaults to PR base or `main`).
   * **`experiment_ref`**: Git ref for experiment (defaults to current commit).
   * **`benchmark_filter`**: Regex to filter benchmarks by name (e.g. `^h2d_.*`).
   * **`environment_filter`**: Regex to filter by environment configuration ID.
   * **`custom_metadata`**: JSON string of metadata to append to the result.

---

## Adding a New Benchmark or Hardware Platform

### Step 1: Implement the Workload & Metrics Emission

1. Create your benchmark script or binary in `tpu_sync/benchmarks/` (or under
   `examples/microbenchmarks/`).
2. Use [`perf_core.py`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/perf_core.py) for standard transfer loops and timing.
3. Emit metrics using [`bap_metrics.py`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/bap_metrics.py):

   ```python
   from tpu_sync.benchmarks import bap_metrics

   # Record metric values (single float or list of floats across steps)
   bap_metrics.emit({
       "my_metric/d2h_gbps": d2h_bandwidth_gbps,
       "my_metric/h2d_gbps": h2d_bandwidth_gbps,
   })
   ```

### Step 2: Define the Benchmark in `benchmark_registry.pbtxt`

Add a new `benchmarks { ... }` block to
[`tpu_sync/benchmarks/benchmark_registry.pbtxt`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/benchmark_registry.pbtxt):

```protobuf
benchmarks {
  name: "my_new_benchmark"
  description: "Description of the new benchmark workload."
  owner: "raiden-dev"

  workload {
    action: "./ml_actions/actions/workload_executors/bazel"
    action_inputs {
      key: "target"
      value: "//tpu_sync/benchmarks:my_new_benchmark"
    }
    action_inputs {
      key: "runtime_flags"
      value: "--iters=10"
    }
  }

  environment_configs {
    id: "tpu-v5e-single-node"
    runner_label: "linux-x86-ct5lp-224-8tpu"
    container_image: "us-central1-docker.pkg.dev/tpu-prod-env-multipod/tpu-raiden/ml-build:gcc12-headers"
    workload_action_inputs {
      key: "bazel_run_flags"
      value: "-c opt --config=oss --config=ci"
    }
  }

  metrics {
    name: "my_metric/d2h_gbps"
    unit: "Gbps"
    stats { stat: MEAN }
  }
}
```

---

## Removing a Benchmark

1. **Update the Registry**: Remove the `benchmarks { ... }` block from
   [`benchmark_registry.pbtxt`](https://github.com/google/tpu-sync/blob/main/tpu_sync/benchmarks/benchmark_registry.pbtxt).
2. **Commit and Merge**: BAP will immediately stop executing the workload.
