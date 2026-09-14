# Features, Limitations and Roadmap

A precise account of what the engine can do today, what it cannot, and what the obvious next steps are.

## Implemented features

### Core network

- [x] Fully connected feed-forward network with any number of layers (`create_network(n)` + `create_layer` per slot).
- [x] Per-layer choice of activation function via function pointers.
- [x] ReLU, sigmoid (numerically clamped) and linear activations, each with its derivative.
- [x] Uniform random weight initialisation in [-0.5, 0.5), zero bias initialisation.
- [x] Forward pass with cached inputs, pre-activations and activations.
- [x] Full backpropagation (`dZ`, `dW`, `db`, `dA_prev`) through every layer.
- [x] Mean squared error gradient for regression targets.

### Optimisation

- [x] Mini-batch gradient descent with configurable batch size.
- [x] Ragged last batch handled without reallocation (column count is shrunk temporarily).
- [x] Plain SGD optimizer.
- [x] Adam optimizer with bias correction and per-layer moment caches.
- [x] Pluggable optimizer interface (`Optimizer.update_func`), so new optimizers need no changes elsewhere.

### Data handling

- [x] Two-pass CSV loader: measure dimensions, allocate once, then parse.
- [x] Optional header row.
- [x] Arbitrary number of input columns and target columns at load time.
- [x] Train / test / validation split by ratio (sequential, zero-copy pointer views).
- [x] Min-max normalisation of inputs fitted on the training split only, applied to every split.
- [x] Target normalisation with stored min/max, and de-normalisation of predictions so metrics are in real units.
- [x] Batch packing into a preallocated `(features × batch)` matrix.

### Training and evaluation loop

- [x] Per-epoch training MSE and MAE in de-normalised units.
- [x] Per-epoch validation pass (forward only).
- [x] Final blind test on the held-out test split.
- [x] Per-batch cache freeing so memory use is flat across epochs.

### Performance

- [x] POSIX thread pool with a ring-buffer task queue, producer/consumer condition variables, graceful shutdown and failure cleanup.
- [x] Row-partitioned parallel matrix multiplication (`dot_product`) with a single-threaded fallback (`pool == NULL`).
- [x] Zero-allocation batch buffers reused across all batches and epochs.
- [x] Measured ~96x speedup over single-sample SGD across 10 epochs (see [mini_batching_results.md](../mini_batching_results.md)).

### Instrumentation

- [x] Monotonic-clock per-epoch timer.
- [x] CSV benchmark log (`epoch,epoch_time_ms,total_time_ms,rmse_ms,mae_sec`), flushed after every epoch.
- [x] Console progress lines for every epoch.

### Data pipeline (Python)

- [x] Join of the raw F1 tables into a per-lap dataset.
- [x] Engineered `stint` and `tire_age` features from pit-stop data.
- [x] Per-race outlier filtering at 120% of the median lap time.
- [x] Final 550,206-row dataset committed to the repository, so no download is needed to run.

## Default experiment and reference results

| Setting | Value |
| --- | --- |
| Model | 7 → 64 (ReLU) → 32 (ReLU) → 1 (linear) |
| Optimizer | Adam, learning rate 0.05 |
| Batch size / epochs | 1024 / 100 |
| Threads | 12 |
| Split | 70 / 20 / 10 train / test / val, chronological |

Reference run on a 10-core Apple Silicon laptop with `-O2`:

| Metric | Value |
| --- | --- |
| Time per epoch | ~1.0 s |
| Total for 100 epochs | ~101 s |
| Train MAE after epoch 1 | ~10,880 ms |
| Train MAE from epoch ~15 onward | ~10,480 ms (plateau) |
| Validation MAE at the plateau | ~13,160 ms |
| Blind test MAE | ~13,925 ms (about 13.9 s per lap) |

The model converges quickly to a plateau and does not improve after roughly fifteen epochs with the default hyperparameters. Given that five of the seven inputs are categorical IDs treated as continuous values, this is expected; see the limitations below for the levers that are most likely to move these numbers.

## Known limitations and quirks

### Modelling

- **Categorical IDs as numbers.** `raceId`, `circuitId`, `constructorId`, `driverId` and `year` are integer identifiers, min-max scaled and fed straight into the network. Neighbouring IDs have no real relationship, so the network cannot learn much from them. One-hot encoding or embeddings would be the standard fix.
- **No shuffling.** The split is sequential and batches are drawn in file order on every epoch. Gradient noise is therefore correlated across epochs.
- **Single target column.** `normalize_dataset` and the metric code assume exactly one target.
- **Deterministic initialisation.** `rand()` is never seeded, so every run starts from identical weights. Good for reproducibility, but there is no way to try a different seed without editing code.
- **No regularisation, dropout, batch norm, learning-rate schedule or early stopping.**
- **No classification support.** Only the MSE gradient exists; there is no cross-entropy or softmax.
- **No model persistence.** Weights cannot be saved to or loaded from disk, and there is no inference-only entry point.

### Reporting

- **Unit labels say seconds, values are milliseconds.** The target is `milliseconds`, and errors are de-normalised into the same unit. The console string "seconds off per lap" and the CSV column `mae_sec` are therefore mislabeled by a factor of 1000. Read every reported MAE and RMSE as milliseconds.
- **Validation metrics are printed but not logged.** `training_benchmark.csv` records the training MSE/MAE only; validation MAE appears on the console line printed by `main.c`.
- **Epoch time includes validation.** The logger's timer runs from the start of training batches to the end of the validation pass.

### Engineering

- **No build system.** The compile command must be typed (or copied from the README). A `Makefile` would be a small, high-value addition.
- **No automated tests.** Correctness relies on `assert` checks in the matrix code and on manual inspection of the loss curve.
- **Hardcoded configuration.** Dataset path, architecture, epochs, batch size, learning rate and thread count are literals in `main()`; there is no CLI or config file.
- **Fixed 4096-byte line buffer** in `load_csv`; longer lines are silently split.
- **`thread_pool_submit` return value is unchecked** in `dot_product`. This is safe with the default queue capacity (1024) because at most `num_threads` tasks are ever queued at once, but a much smaller capacity would silently drop work.
- **Allocation-heavy inner loop.** Every matrix operation allocates a fresh result, so the forward/backward pass performs several `malloc`/`free` pairs per layer per batch. Batching itself is allocation-free; the layer math is not yet.
- **Windows artefact in the repo.** `neural_network.exe` is a committed MinGW build and will go stale as the code changes.
- **`.DS_Store` is tracked** and should be ignored.

## Roadmap ideas

Ordered roughly by expected value for effort.

1. **Makefile / CMake** with `all`, `clean` and an `-O2` default.
2. **Fix the unit labels** ("ms" in the console strings and a `mae_ms` CSV column) and log validation metrics alongside training metrics.
3. **Epoch-level shuffling** of the training index array (only the pointer views need permuting).
4. **Categorical encoding** in `cleaner.py` (one-hot for `circuitId` and `constructorId`, or drop `raceId`) so the network receives meaningful inputs.
5. **Seedable initialisation** (`srand`) and Xavier/He initialisation options.
6. **Learning-rate schedule and early stopping** on validation MAE.
7. **Save / load weights** to a binary file, plus an inference-only mode.
8. **In-place matrix operations** for the hot path to remove per-batch allocations.
9. **Cross-entropy loss and softmax** to support classification.
10. **Unit tests** for the matrix library and thread pool (gradient checking against finite differences would catch most backprop regressions).
11. **Command-line arguments** for epochs, batch size, learning rate, threads and dataset path.

## Development history

Reconstructed from the git log:

| Milestone | Commit message |
| --- | --- |
| Core modules complete, `main` pending | "need to write the main function, rest is done" |
| Thread pool wired into the network | "added thread_pool to nn" |
| Adam optimizer | "added adam" |
| CSV data loader | "data loader" |
| Mini-batch training | "added mini-batching, this is a real game changer guys" |
