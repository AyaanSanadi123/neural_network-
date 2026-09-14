# Getting Started

This guide takes a new user from a fresh clone to a completed training run, then shows how to change the experiment and how to train on a different dataset.

## 1. Prerequisites

| Requirement | Why | Notes |
| --- | --- | --- |
| C compiler with C99 and pthreads | Building the engine | macOS: Xcode Command Line Tools (`xcode-select --install`). Linux: `gcc` or `clang`. Windows: MinGW-w64 (for example via MSYS2). MSVC is not supported because the code uses `pthread.h`, `unistd.h` and `clock_gettime`. |
| ~40 MB free disk | The repository ships the raw F1 CSVs (21 MB) and the final dataset (15 MB) | Nothing needs to be downloaded separately to run the default experiment. |
| Python 3.9+ with `pandas` | Only if you regenerate the dataset | See [DATA_PIPELINE.md](DATA_PIPELINE.md). |

## 2. Clone

```bash
git clone https://github.com/AyaanSanadi123/neural_network-.git
cd neural_network-
```

## 3. Build

There is no Makefile or CMake project yet. Every module lives in its own folder and headers are included by bare name (for example `#include "matrix.h"`), so each folder must be passed as an include path.

### macOS and Linux

```bash
gcc -O2 -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool -Ibenchmarking_engine -Idata_preprocessing main.c activation_functions/activations.c layers/layer.c loss_functions/losses.c matrix/matrix.c network/fp.c optimizer/optimizer.c thread_pool/threadpool.c benchmarking_engine/b_engine.c data_preprocessing/data_loader.c -o neural_network -pthread -lm
```

`clang` works with the identical command (on macOS `gcc` is an alias for Apple clang). `-O2` is optional but makes epochs several times faster.

### Windows (MinGW-w64)

This is the command recorded at the bottom of [main.c](../main.c), which is what produced the committed `neural_network.exe`:

```bash
gcc -Iactivation_functions -Ilayers -Iloss_functions -Imatrix -Inetwork -Ioptimizer -Ithread_pool -Ibenchmarking_engine -Idata_preprocessing main.c activation_functions\activations.c layers\layer.c loss_functions\losses.c matrix\matrix.c network\fp.c optimizer\optimizer.c thread_pool\threadpool.c benchmarking_engine\b_engine.c data_preprocessing\data_loader.c -o neural_network.exe -pthread -lm
```

MinGW-w64 provides `pthread.h` (winpthreads), `unistd.h` and `clock_gettime`, which the code relies on.

### What gets compiled

| Source file | Module |
| --- | --- |
| `main.c` | Training loop, test loop, `main()` configuration |
| `matrix/matrix.c` | Matrix type and operations |
| `layers/layer.c` | Dense layer |
| `activation_functions/activations.c` | Activation functions |
| `loss_functions/losses.c` | Loss gradient |
| `network/fp.c` | Forward and backward passes |
| `optimizer/optimizer.c` | SGD and Adam |
| `thread_pool/threadpool.c` | Thread pool |
| `benchmarking_engine/b_engine.c` | Benchmark logger |
| `data_preprocessing/data_loader.c` | CSV loading, splitting, normalisation, batching |

## 4. Run

Run from the repository root. The dataset path inside `main()` is relative (`data/lap_time_predictor_dataset_cleaned.csv`), so running from another directory fails with `Fetal error, could not open file`.

```bash
./neural_network
```

On Windows:

```bash
neural_network.exe
```

The program:

1. Scans the CSV once to count rows and columns.
2. Allocates one small matrix per sample and loads the file.
3. Splits the data 70% train / 20% test / 10% validation in file order.
4. Fits min-max normalisation on the training split and applies it to all splits (inputs and target).
5. Starts a 12-thread pool and the benchmark logger.
6. Builds a 7 → 64 → 32 → 1 network and an Adam optimizer.
7. Trains for 100 epochs of 1024-sample mini-batches, validating after every epoch.
8. Runs a final forward-only pass over the test split and prints the blind-test error.
9. Frees everything and exits.

### Expected runtime

| Machine | Time per epoch |
| --- | --- |
| 10-core Apple Silicon laptop, `-O2`, 12 threads | ~1.0 s |
| Original benchmark machine (see [mini_batching_results.md](../mini_batching_results.md)) | ~3.7 s |

A full 100-epoch run therefore takes roughly 2 to 7 minutes plus a few seconds of loading.

### Reading the output

Console output looks like this (numbers vary by machine):

```
Scanning dataset dimensions...
Found 550206 rows, 8 columns.
Loading CSV into memory arenas...
--- STARTING TRAINING LOOP (100 Epochs, 385144 Samples, Batch Size: 1024) ---
Epoch 1 | Train MAE: 10879.6723 | Val MAE: 12183.3416
[Epoch   1] Time:   976.33 ms | Total:   976.33 ms | MAE: 10879.672 s
Epoch 2 | Train MAE: 10293.4717 | Val MAE: 12728.9493
[Epoch   2] Time:   989.10 ms | Total:  1965.42 ms | MAE: 10293.472 s
...
--- INITIATING FINAL BLIND TEST PHASE ---
=========================================
FINAL ENGINE GRADE ON UNSEEN TRACK DATA:
Mean Squared Error: ...
Mean Absolute Error: ... seconds off per lap
=========================================
--- CLEANING UP ---
--- SHUTDOWN SUCCESSFUL ---
```

Two lines are printed per epoch: one from the training loop in `main.c` (train and validation MAE) and one from the benchmark logger (timing and train MAE).

**Units:** the target column is `milliseconds`, and errors are converted back to real units before being averaged, so MAE and RMSE are in **milliseconds** even though the console labels and the `mae_sec` CSV column say seconds. An MAE of 10,000 means the model is about 10 seconds off per lap.

`training_benchmark.csv` is rewritten on every run in the current directory:

| Column | Meaning |
| --- | --- |
| `epoch` | 1-based epoch number |
| `epoch_time_ms` | Wall-clock time of that epoch (training + validation), monotonic clock |
| `total_time_ms` | Cumulative time so far |
| `rmse_ms` | Square root of the training MSE, in target units (milliseconds) |
| `mae_sec` | Training MAE, in target units (milliseconds, despite the name) |

## 5. Changing the experiment

All knobs are plain values in `main()` in [main.c](../main.c). Edit, rebuild, rerun.

| What | Where in `main()` | Default |
| --- | --- | --- |
| Dataset file | `dataset_file` | `data/lap_time_predictor_dataset_cleaned.csv` |
| Input / target column counts | `create_dataset(rows, 7, 1)` | 7 inputs, 1 target |
| Header row present | third argument of `count_csv_dimensions` and `load_csv` | `1` (yes) |
| Split ratios | `split_dataset_sequential(dataset, 0.7, 0.2, 0.1)` | train, **test**, val (note the order) |
| Benchmark output file | `logger_init("training_benchmark.csv")` | `training_benchmark.csv` |
| Worker threads and queue size | `thread_pool_init(12, 1024)` | 12 threads, 1024 queued tasks |
| Layers | `create_network(3)` and the three `create_layer` calls | 7→64 ReLU, 64→32 ReLU, 32→1 linear |
| Optimizer | `create_adam_optimizer(0.05, ...)` | Adam, lr 0.05 |
| Epochs | `epochs` | 100 |
| Batch size | `batch_size` | 1024 |

Tips:

- Set the thread count to the number of physical cores on your machine. More threads than cores adds contention without speed.
- To use plain SGD instead of Adam, replace the optimizer line with `create_sgd_optimizer(learning_rate)`. Both optimizers share the `Optimizer` interface, so nothing else changes.
- Each layer's activation is chosen per layer. Use `sigmoid`/`sigmoid_derivative` or `linear`/`linear_derivative` from `activations.h` in place of the ReLU pair.
- The number passed to `create_network` must equal the number of `create_layer` assignments, and each layer's input size must equal the previous layer's output size.

## 6. Training on your own dataset

The loader is generic. Your CSV must satisfy these rules:

1. **Numeric only.** Every cell is parsed with `atof`. Strings, quoted fields and dates are not supported (empty cells become `0.0`).
2. **Comma separated**, one sample per line, lines shorter than 4096 bytes.
3. **Input columns first, then target columns.** With `create_dataset(rows, N, 1)` the first `N` columns are inputs and column `N+1` is the target.
4. **One target column.** Normalisation and the metrics assume a single regression target.
5. An optional header row, in which case pass `1` as `has_header`.

Then in `main()`:

1. Point `dataset_file` at your file.
2. Change `create_dataset(rows, 7, 1)` to your input count.
3. Change the first layer's input size (`create_layer(7, 64, ...)`) to match.
4. Rebuild and run.

The split is sequential, not shuffled, so if your file is sorted by something meaningful (time, class, etc.) the train, test and validation sets will differ in distribution. Shuffle the file beforehand if that is not what you want.

## 7. Regenerating the F1 dataset

The final CSV is committed, so this is optional. The two pandas scripts in `data/` rebuild it from the raw tables in `data/f1_data/`:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install pandas
cd data
python cleaner.py     # joins tables, engineers tire_age -> lap_time_predictor_dataset.csv
python cleaner1.py    # drops outlier laps -> lap_time_predictor_dataset_cleaned.csv
```

Both scripts use paths relative to the `data/` folder, so run them from there. Details are in [DATA_PIPELINE.md](DATA_PIPELINE.md).

## 8. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `matrix.h: No such file or directory` | A `-I<folder>` flag is missing. Copy the full build command above. |
| `undefined reference to pthread_create` | Add `-pthread`. |
| `undefined reference to sqrt`/`exp`/`pow` | Add `-lm` at the end of the command. |
| `Fetal error, could not open file data/...` | You are not running from the repository root, or the dataset is missing. |
| `Please enter a valid number of threads or queue size` | `thread_pool_init` was called with a non-positive value. |
| Very slow epochs | Build with `-O2`; check the thread count matches your core count. |
| Errors reported as huge numbers | They are in milliseconds (lap times are ~90,000 ms). See the units note above. |
| Program exits after loading with a crash | The declared input count in `create_dataset` does not match the CSV, or a layer's input size does not match the previous output size. Assertions in the matrix code will abort with a message. |
