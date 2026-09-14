# Project Structure

The repository is organised as one folder per engine module, each containing a header (`.h`) that declares the public interface and a source file (`.c`) that implements it. `main.c` at the root wires the modules together into a training run.

```
neural_network-/
├── main.c                          # Training loop, test loop, main() with the experiment config
├── README.md                       # Project overview and quick start
├── CONTRIBUTING.md                 # How to extend the engine
├── mini_batching_results.md        # Benchmark write-up: single-sample SGD vs mini-batching
├── training_benchmark.csv          # Per-epoch metrics from the most recent run (regenerated each run)
├── neural_network.exe              # Prebuilt Windows x86-64 binary (MinGW build)
├── .gitignore                      # Ignores the Python virtualenv (myenv/)
│
├── matrix/                         # Matrix type + linear algebra
│   ├── matrix.h
│   └── matrix.c
├── layers/                         # Dense (fully connected) layer
│   ├── layer.h
│   └── layer.c
├── activation_functions/           # Element-wise activations and derivatives
│   ├── activations.h
│   └── activations.c
├── loss_functions/                 # Loss gradients
│   ├── losses.h
│   └── losses.c
├── network/                        # Network container, forward pass, backpropagation
│   ├── fp.h
│   └── fp.c
├── optimizer/                      # SGD and Adam
│   ├── optimizer.h
│   └── optimizer.c
├── thread_pool/                    # POSIX thread pool
│   ├── threadpool.h
│   └── threadpool.c
├── benchmarking_engine/            # Epoch timer + CSV metrics logger
│   ├── b_engine.h
│   └── b_engine.c
├── data_preprocessing/             # CSV loading, splitting, normalisation, batching
│   ├── data_loader.h
│   └── data_loader.c
│
├── data/                           # Dataset and the scripts that build it
│   ├── f1_data/                    # Raw Formula 1 tables (14 CSVs, ~21 MB)
│   ├── cleaner.py                  # Step 1: join tables, engineer features
│   ├── cleaner1.py                 # Step 2: remove outlier laps
│   └── lap_time_predictor_dataset_cleaned.csv   # Final training data (550,206 rows)
│
├── results/                        # Archived benchmark outputs
│   └── training_benchmark_before_mini_batching.csv
│
├── docs/                           # This documentation
└── myenv/                          # Python virtualenv (git-ignored, local only)
```

## Module folders

### `matrix/`

The foundation of the engine. Defines the `Matrix` struct (row count, column count, one contiguous `double` array in row-major order) and every linear-algebra operation the network needs.

| File | Contents |
| --- | --- |
| `matrix.h` | `Matrix` struct and 12 operation prototypes. |
| `matrix.c` | Constructor/destructor, the thread-pooled `dot_product`, transpose, Hadamard product, copy, add, subtract, scalar multiply, element-wise map, bias broadcast, and row-wise column sum. |

`dot_product` is the only threaded operation: it splits the rows of the left operand across the pool's workers. Passing `NULL` as the pool runs it single-threaded.

### `layers/`

One dense layer. Holds the trainable parameters, the caches saved during the forward pass for use in backpropagation, the gradient buffers, and function pointers to the layer's activation and its derivative.

| File | Contents |
| --- | --- |
| `layer.h` | `layer` struct, `create_layer`, `layer_free_caches`, `free_layer`, `random_uniform`. |
| `layer.c` | Uniform weight initialisation in [-0.5, 0.5), zero biases, cache management. |

### `activation_functions/`

Scalar functions applied element-wise by `matrix_map`. Each activation ships with its derivative so a layer can be created with a matching pair.

| Function | Derivative |
| --- | --- |
| `relu` | `relu_derivative` |
| `sigmoid` (clamped to 0/1 beyond ±6) | `sigmoid_derivative` |
| `linear` | `linear_derivative` |

### `loss_functions/`

Gradients of loss functions with respect to the network output. Currently one function, `mse_derivative`, which returns `(prediction - expected) / batch_size` element-wise. The loss value itself is not computed here; `main.c` accumulates MSE and MAE directly.

### `network/`

The `network` container (an array of layer pointers) and the algorithms that run over it. Despite the file name `fp.c` ("forward pass"), this module holds the full training step: forward pass with caching, backpropagation through every layer, the optimizer dispatch loop, and cache/network teardown.

### `optimizer/`

A small polymorphic optimizer interface. `Optimizer` holds a learning rate, an opaque `state` pointer, and an `update_func` callback invoked once per layer per batch.

| Optimizer | State | Notes |
| --- | --- | --- |
| SGD (`create_sgd_optimizer`) | none | `w -= lr * dw` |
| Adam (`create_adam_optimizer`) | `AdamState` with per-layer first/second moment matrices | β1 = 0.9, β2 = 0.999, ε = 1e-8, bias-corrected, timestep advanced once per batch |

### `thread_pool/`

A classic fixed-size POSIX thread pool: N worker threads, a ring-buffer task queue, one mutex, and two condition variables (`notify` wakes workers when work arrives; `all_idle` wakes the submitter when every task has finished). Used by `dot_product` to parallelise matrix multiplication.

### `benchmarking_engine/`

`BenchLogger` opens a CSV, timestamps the start of each epoch with `clock_gettime(CLOCK_MONOTONIC)`, and on epoch end writes epoch time, cumulative time, RMSE and MAE, flushing after each line so partial results survive a crash.

### `data_preprocessing/`

Everything between a CSV on disk and a batch matrix in the network:

| Function | Role |
| --- | --- |
| `count_csv_dimensions` | Pass 1: count rows and columns with a 64 KB buffered read. |
| `create_dataset` | Allocate one input matrix and one target matrix per sample (the "memory arena"). |
| `load_csv` | Pass 2: parse every line with `strchr`/`atof` into the arena. |
| `split_dataset_sequential` | Build train/test/val pointer views over the arena (no copying, no shuffling). |
| `normalize_dataset` | Min-max scale every input column and the target, using statistics from the training split only. |
| `get_batch` | Copy a contiguous run of samples into a preallocated `(features × batch)` matrix. |

## Data and results folders

### `data/`

| Path | Description |
| --- | --- |
| `f1_data/` | Raw Formula 1 archive: `circuits`, `constructors`, `constructor_results`, `constructor_standings`, `drivers`, `driver_standings`, `lap_times` (589k rows), `pit_stops`, `qualifying`, `races`, `results`, `seasons`, `sprint_results`, `status`. Only `lap_times`, `pit_stops`, `races` and `results` are used by the pipeline. |
| `cleaner.py` | Joins lap times with race year/circuit and driver constructor, derives pit-stop stints and `tire_age`, writes `lap_time_predictor_dataset.csv` (intermediate, not committed). |
| `cleaner1.py` | Drops laps slower than 120% of their race's median lap time (safety cars, pit-lane crawls), writes `lap_time_predictor_dataset_cleaned.csv`. |
| `lap_time_predictor_dataset_cleaned.csv` | Final dataset. Columns: `raceId, circuitId, constructorId, driverId, year, lap, tire_age, milliseconds`. 550,206 samples. |

### `results/`

Archived benchmark CSVs kept for comparison. `training_benchmark_before_mini_batching.csv` uses the same column layout as `training_benchmark.csv`. Note that the committed copies currently contain only the header row; the actual numbers from that experiment are tabulated in [mini_batching_results.md](../mini_batching_results.md).

## Root-level files

| File | Description |
| --- | --- |
| `main.c` | `train_network`, `test_network`, and `main()`. This is the only place that decides dataset, architecture, optimizer, epochs and batch size. The Windows build command is preserved as a comment at the bottom. |
| `training_benchmark.csv` | Output of the last training run. Overwritten every run. |
| `neural_network.exe` | Windows binary produced by the command in `main.c`. It must be run from the repository root so it can find `data/`. Not usable on macOS/Linux; rebuild instead. |
| `mini_batching_results.md` | Empirical comparison of single-sample SGD versus 1024-sample mini-batches over 10 epochs. |
| `.gitignore` | Excludes `myenv/`. |
| `myenv/` | Local Python 3.14 virtualenv containing pandas, numpy and the Kaggle CLI. Not tracked by git; create your own (see [GETTING_STARTED.md](GETTING_STARTED.md#7-regenerating-the-f1-dataset)). |

## Dependency graph between modules

```
main.c
 ├── data_loader   ──► matrix
 ├── b_engine
 ├── threadpool
 ├── activations
 ├── losses        ──► matrix
 ├── optimizer     ──► layer ──► matrix ──► threadpool
 └── fp (network)  ──► layer, losses, optimizer, threadpool
```

`matrix.h` includes `threadpool.h` because `dot_product` takes a pool argument, which is why the thread pool sits at the bottom of the graph. `activations` and `b_engine` are leaf modules with no internal dependencies.
