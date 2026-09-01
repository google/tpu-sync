# Raiden Microbenchmarks

## Overview

This folder contains microbenchmarks for the Raiden engine. They differ in which
part of the transfer path they isolate:

| Benchmark | Measures | Needs TPU? | Framework |
| --- | --- | --- | --- |
| `jax_dma_kv_cache_benchmark.py` | Raw DMA on ONE host: D2H and H2D, against JAX baselines | yes | JAX |
| `torch_dma_kv_cache_benchmark.py` | Raw DMA on ONE host: D2H and H2D, against PyTorch baselines | yes | PyTorch |
| `jax_d2d_read_benchmark_runner.py` | Cross-node device-to-device pull: D2H + H2H + H2D end to end | yes, on both nodes | JAX |
| `h2h_benchmark_runner.cc` | The middle hop only: host memory to host memory across the NIC | no | None (C++) |

Pick by what you are trying to localise. The C++ H2H runner gives the wire
ceiling; the D2D read runner shows what the device path delivers against that
ceiling; the DMA benchmark shows whether the device copies themselves are the
limit. See [H2H.md](H2H.md) for the C++ runner.

---

## `jax_dma_kv_cache_benchmark.py` — single-host raw DMA

**Prerequisite:** Ensure you have already installed the package in your environment before running these scripts.

To execute the microbenchmark, navigate to this directory (`examples/microbenchmarks/`) and run:

```bash
PYTHONPATH=../.. python jax_dma_kv_cache_benchmark.py --telemetry_log_path=/tmp/${USER}_benchmark.jsonl
```

*Note: Setting PYTHONPATH=../.. points the Python interpreter back to the repository root so it can discover the tpu_raiden package, and the `--telemetry_log_path` flag prevents permission errors when writing output on shared VMs.*

### How to Read the Output

The microbenchmark will run through a suite of test cases (varying data types like BF16, FP32, INT32, and different tensor shapes).

For each test case, the standard output will display a performance comparison between three implementations:

1. **KVCacheManager**: The TPU Raiden raw DMA engine.
2. **JAX Pinned Host Baseline**: Native JAX transfers using pinned host memory.
3. **JAX Standard Baseline**: Native JAX transfers using standard unpinned NumPy arrays.

The script prints the median latency (in seconds) and the calculated throughput (in GB/s) for both **D2H (Device-to-Host)** and **H2D (Host-to-Device)** transfers.

When evaluating the performance, look specifically at the **`KVCacheManager D2H bandwidth`** and **`KVCacheManager H2D bandwidth`** lines and compare them against the JAX baselines to observe the throughput gains achieved by bypassing the framework overhead.

---

## `torch_dma_kv_cache_benchmark.py` — single-host raw DMA (PyTorch)

To execute the PyTorch microbenchmark on physical TPU hardware or via Forge:

```bash
blaze test --test_strategy=forge //third_party/tpu_raiden/examples/microbenchmarks:torch_dma_kv_cache_benchmark_test_gl
```

Or locally on a TPU VM:

```bash
PYTHONPATH=../.. python torch_dma_kv_cache_benchmark.py --telemetry_log_path=/tmp/${USER}_torch_benchmark.jsonl
```

### How to Read the Output

The microbenchmark runs across model geometries, layers, and dtypes (`bf16`, `fp32`, `int32`), comparing three implementations:

1. **KVCacheManager**: The TPU Raiden raw DMA engine.
2. **PyTorch Pinned Host Baseline**: Native PyTorch transfers using page-locked host memory (`pin_memory=True`, `non_blocking=True`).
3. **PyTorch Standard Baseline**: Native PyTorch transfers using standard unpinned host memory.

The script logs telemetry with 95% confidence intervals and standard deviation to `--telemetry_log_path`, recording median latency and throughput (GB/s) for both D2H and H2D.


