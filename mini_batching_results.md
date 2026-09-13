# Performance Benchmark: 96x Acceleration via Zero-Allocation Mini-Batching & Thread Pooling

This document details the architectural bottleneck analysis, systems-level refactor, and empirical performance benchmarks for the custom C neural network engine predicting Formula 1 lap times.

---

## 1. Overview & Core Achievement

Transitioning from an unbatched, single-sample Stochastic Gradient Descent (SGD) pipeline to a zero-allocation, multithreaded mini-batch architecture yielded a **95.92x speedup** across 10 training epochs on a 550,000-sample telemetry dataset.

* **Single-Sample SGD Runtime:** 3,525.74 seconds (~58.76 minutes)
* **Mini-Batch (B=1024) Runtime:** 36.76 seconds (~0.61 minutes)
* **Latency Reduction:** **98.96%**
* **Hardware Utilization:** 12-thread POSIX thread pool sustained at near-100% saturation.

---

## 2. Benchmark Comparison

All benchmarks were captured using high-resolution monotonic clocks (`clock_gettime(CLOCK_MONOTONIC)`) via `BenchLogger` on a dataset of ~550,000 samples across 10 epochs.

### Runtime Metrics

| Epoch | Single-Sample SGD Latency | Mini-Batch (B=1024) Latency | Speedup Factor |
| :---: | :---: | :---: | :---: |
| **1** | 255.67 s | 3.66 s | 69.85x |
| **2** | 274.79 s | 3.66 s | 75.08x |
| **3** | 337.83 s | 3.62 s | 93.32x |
| **4** | 355.71 s | 3.89 s | 91.44x |
| **5** | 362.01 s | 3.67 s | 98.64x |
| **6** | 361.32 s | 3.64 s | 99.26x |
| **7** | 360.68 s | 3.65 s | 98.82x |
| **8** | 397.29 s | 3.69 s | 107.67x |
| **9** | 395.69 s | 3.63 s | 108.99x |
| **10** | 424.75 s | 3.66 s | 116.05x |
| **Total** | **3,525.74 s (~58.76 min)** | **36.76 s (~0.61 min)** | **95.92x** |
| **Average / Epoch** | **352.57 s** | **3.68 s** | **95.92x** |