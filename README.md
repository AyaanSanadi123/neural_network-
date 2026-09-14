# Neural Network Engine in C: Formula 1 Lap Time Predictor

A feed-forward neural network written from scratch in C with no machine learning libraries. It trains a regression model that predicts Formula 1 lap times from race, circuit, constructor, driver, season, lap number and tire age. Everything underneath is hand-built: the matrix library, the layers, backpropagation, the Adam optimizer, a POSIX thread pool that parallelises matrix multiplication, a CSV data loader with min-max normalisation, mini-batch training and a benchmark logger.

The training data is a 550k-row dataset engineered from the public Formula 1 results archive (Ergast-derived CSVs distributed on Kaggle) using the Python scripts in `data/`.

## Features available today

| Area | What you get |
| --- | --- |
| **Network** | Arbitrary number of fully connected layers, each with its own activation function. |
| **Activations** | ReLU, sigmoid (clamped for numerical safety), linear, with derivatives. |
| **Loss** | Mean squared error gradient for regression. MSE and MAE are reported in the target's real units. |
| **Optimizers** | Plain SGD and Adam (bias-corrected, per-layer moment caches). |
| **Training** | Mini-batch gradient descent with a zero-allocation batch buffer, ragged last batch handling, per-epoch validation and a final blind test on held-out data. |
| **Parallelism** | POSIX thread pool (ring-buffer task queue, producer/consumer with condition variables) used by the matrix dot product. |
| **Data loading** | Two-pass CSV loader (measure, then fill preallocated arenas), sequential train/test/val split, min-max normalisation fitted on the training split only, batch packing. |
| **Benchmarking** | Monotonic-clock timing per epoch written to `training_benchmark.csv` alongside RMSE and MAE. |
| **Data pipeline** | Pandas scripts that join the raw F1 tables, engineer a `tire_age` feature from pit stops, and remove safety-car outliers. |

See [docs/FEATURES.md](docs/FEATURES.md) for the full list, known limitations and roadmap.

## Quick start

You need a C compiler with pthreads (Xcode Command Line Tools on macOS, gcc on Linux, MinGW-w64 on Windows). Python is only needed if you want to regenerate the dataset.

```bash
git clone https://github.com/AyaanSanadi123/neural_network-.git
cd neural_network-
```

Build (macOS / Linux):

```bash
gcc -O2 -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool -Ibenchmarking_engine -Idata_preprocessing main.c activation_functions/activations.c layers/layer.c loss_functions/losses.c matrix/matrix.c network/fp.c optimizer/optimizer.c thread_pool/threadpool.c benchmarking_engine/b_engine.c data_preprocessing/data_loader.c -o neural_network -pthread -lm
```

Run from the repository root (the dataset path is relative):

```bash
./neural_network
```

You will see the dataset being scanned and loaded, then one line per epoch with train and validation MAE, and finally the blind-test score. Timing and error metrics are written to `training_benchmark.csv`.

Full instructions, including Windows and how to change the model, live in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Default experiment

The `main()` in [main.c](main.c) runs this configuration:

| Setting | Value |
| --- | --- |
| Dataset | `data/lap_time_predictor_dataset_cleaned.csv` (550,206 rows, 7 inputs, 1 target) |
| Split | 70% train / 20% test / 10% validation, in file order |
| Architecture | 7 → 64 (ReLU) → 32 (ReLU) → 1 (linear) |
| Optimizer | Adam, learning rate 0.05 |
| Batch size | 1024 |
| Epochs | 100 |
| Thread pool | 12 worker threads, queue capacity 1024 |

Switching from single-sample SGD to this mini-batched, thread-pooled pipeline produced a ~96x speedup over 10 epochs (58.8 minutes down to 36.8 seconds). The measurements are in [mini_batching_results.md](mini_batching_results.md).

## Project layout

```
neural_network-/
├── main.c                  # Training loop, test loop, experiment configuration
├── matrix/                 # Matrix type and operations (threaded dot product)
├── layers/                 # Dense layer: weights, biases, caches, gradients
├── activation_functions/   # ReLU, sigmoid, linear + derivatives
├── loss_functions/         # MSE gradient
├── network/                # Forward pass, backpropagation, weight updates
├── optimizer/              # SGD and Adam
├── thread_pool/            # POSIX thread pool
├── benchmarking_engine/    # Per-epoch timing and metrics logger
├── data_preprocessing/     # CSV loader, splitting, normalisation, batching
├── data/                   # Raw F1 CSVs, cleaning scripts, final dataset
├── results/                # Saved benchmark CSVs
├── docs/                   # Documentation (start here)
└── mini_batching_results.md
```

Every folder is described in [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

## Documentation

| Document | Read it when you want to... |
| --- | --- |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Build, run, tune the experiment, or train on your own CSV. |
| [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) | Know what every folder and file is for. |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Understand how data flows through the engine and the maths behind each step. |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Look up a function, struct, or ownership rule. |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | Understand or regenerate the F1 dataset. |
| [docs/FEATURES.md](docs/FEATURES.md) | See what is implemented, what is not, and known quirks. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Add an activation, loss, optimizer, or change the network. |
| [mini_batching_results.md](mini_batching_results.md) | Read the mini-batching benchmark write-up. |

## Status

This is an active learning and research project. There is no build system beyond the gcc command above, no automated tests, and no license file yet. If you plan to reuse the code, ask the author about licensing first.
