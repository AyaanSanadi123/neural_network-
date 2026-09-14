# Architecture

This document explains how the engine works end to end: how data is represented, what happens in a forward and backward pass, how the optimizer updates weights, how the thread pool parallelises the heavy work, and how memory is managed so that a training step allocates as little as possible.

## 1. Big picture

```
 CSV on disk
     │  count_csv_dimensions  (pass 1: rows, cols)
     │  create_dataset        (one (F×1) input + one (T×1) target matrix per sample)
     │  load_csv              (pass 2: parse into the arena)
     ▼
 Dataset arena  ──► split_dataset_sequential ──► train / test / val pointer views
     │
     │  normalize_dataset  (min-max, fitted on train only, applied to all)
     ▼
 ┌──────────────────────────── per epoch ────────────────────────────┐
 │  for each 1024-sample window of the train split:                  │
 │     get_batch          pack samples into X (F×B), Y (T×B)         │
 │     network_forward    A₀=X → Z₁=W₁A₀+b₁ → A₁=g(Z₁) → … → Ŷ       │
 │     metrics            de-normalise Ŷ, Y; accumulate MSE, MAE      │
 │     network_backward   dA=∂L/∂Ŷ → dZ, dW, db per layer, back to A₀ │
 │     network_update_weights   Adam / SGD step on every layer        │
 │     network_free_caches      drop the forward caches               │
 │  validation: forward + metrics only, over the val split           │
 │  logger_log_epoch      time, RMSE, MAE → training_benchmark.csv    │
 └───────────────────────────────────────────────────────────────────┘
     │
     ▼
 test_network   forward-only pass over the test split, prints final MSE / MAE
```

`F` = number of input features (7), `T` = number of targets (1), `B` = batch size (1024).

## 2. Data representation

### The `Matrix` type

```c
typedef struct {
    int cols;
    int rows;
    double* data;   // rows * cols doubles, row-major
} Matrix;
```

Element `(i, j)` lives at `data[i * cols + j]`. Every operation in `matrix.c` returns a **newly allocated** matrix (except `free_matrix`), and the caller owns it. There are no in-place operations, which keeps the code simple at the cost of extra allocations inside the backward pass.

### Samples are columns

A batch is a `(features × batch_size)` matrix: each **column** is one sample, each **row** is one feature. This convention makes the layer equation a single left-multiplication:

```
Z = W · X + b        W: (out × in)   X: (in × B)   b: (out × 1)   Z: (out × B)
```

The bias is a column vector broadcast across every column of `W·X` by `matrix_add_bias`.

### The `Dataset` arena

`create_dataset` allocates, once, an array of `total_samples` small `(F × 1)` input matrices and `(T × 1)` target matrices. The train, test and validation "splits" are just arrays of pointers into that arena, so splitting costs no copies. `get_batch` is the only place samples are copied: it gathers `B` consecutive samples into the batch matrix, one column each.

### Normalisation

`normalize_dataset` computes per-feature min and max **from the training split only**, then applies `(x - min) / (max - min)` to train, test and validation inputs. Columns with zero range become 0. The single target column is scaled the same way and its `target_min` / `target_max` are stored on the `Dataset` so predictions can be mapped back to real units:

```
real = normalised * (target_max - target_min) + target_min
```

`train_network` and `test_network` do this de-normalisation before accumulating MSE and MAE, so reported errors are in the target's original units (milliseconds for the F1 dataset).

## 3. Forward pass

`layer_forward` (in `network/fp.c`) computes one layer and caches everything backpropagation will need:

| Step | Code | Cached as |
| --- | --- | --- |
| Copy the input | `matrix_copy(input)` | `input_cache` (A_prev) |
| Weighted sum | `dot_product(W, X, pool)` | (temporary) |
| Add bias | `matrix_add_bias(WX, b)` | `z_cache` (Z) |
| Activate | `matrix_map(Z, activation_func)` | `activation_cache` (A) |

`network_forward` chains the layers, feeding each layer's `activation_cache` into the next, and returns a **copy** of the last activation so the caller can free it independently of the caches.

Activations are scalar functions of one `double`, applied element-wise via `matrix_map`. The output layer of the default model uses `linear` so the network can emit any real value for regression.

## 4. Backward pass

`network_backward` starts from the loss gradient and walks the layers in reverse. With `L` the loss, `⊙` the Hadamard (element-wise) product and `g'` the activation derivative, `layer_backward` computes, for one layer:

| Quantity | Formula | Code |
| --- | --- | --- |
| dZ | dA ⊙ g'(Z) | `hadamard_product(dA, matrix_map(z_cache, activation_derivative))` |
| dW | dZ · A_prevᵀ | `dot_product(dZ, transpose(input_cache), pool)` |
| db | Σ over the batch of dZ (row sums) | `matrix_sum_columns(dZ)` |
| dA_prev | Wᵀ · dZ | `dot_product(transpose(W), dZ, pool)` |

`dW` and `db` are stored on the layer (`d_weights`, `d_biases`, replacing the previous batch's gradients) and `dA_prev` is returned to become the next layer's `dA`.

The starting gradient comes from `loss_functions/losses.c`:

```
mse_derivative(Ŷ, Y) = (Ŷ - Y) / B
```

The conventional MSE gradient carries a factor of 2; dropping it simply rescales the effective learning rate. Because `B` is read from the matrix's column count, the ragged last batch is averaged correctly.

## 5. Optimizers

The `Optimizer` struct is a tiny interface:

```c
struct Optimizer {
    double learning_rate;
    void*  state;                                        // NULL for stateless optimizers
    void (*update_func)(Optimizer*, layer*, int layer_index);
};
```

`network_update_weights` advances the Adam timestep (if the optimizer has state) and then calls `update_func` once per layer.

### SGD

`w -= lr * dw` and `b -= lr * db`. No state.

### Adam

`create_adam_optimizer` allocates, for every layer, first-moment (`m`) and second-moment (`v`) matrices the same shape as the weights and biases, initialised to zero. Each update does, per parameter with gradient `g`:

```
m = β1·m + (1-β1)·g
v = β2·v + (1-β2)·g²
m̂ = m / (1 - β1ᵗ)
v̂ = v / (1 - β2ᵗ)
w -= lr · m̂ / (√v̂ + ε)
```

with β1 = 0.9, β2 = 0.999, ε = 1e-8. The timestep `t` is incremented once per call to `network_update_weights`, that is once per mini-batch. The bias-correction denominators are computed once per layer per update rather than per parameter.

## 6. Thread pool and parallel matrix multiplication

### The pool

`thread_pool/threadpool.c` implements a fixed-size producer/consumer pool:

- `num_threads` worker threads are created at init and live until `thread_pool_destroy`.
- Tasks (`{ void (*execute)(void*); void* arg; }`) sit in a **ring buffer** of `queue_capacity` slots with `head`, `tail` and `count` indices.
- One mutex (`lock`) protects the queue. Two condition variables coordinate:
  - `notify`: signalled by `thread_pool_submit` when a task is enqueued; workers block on it while the queue is empty.
  - `all_idle`: signalled by a worker when it finishes the last pending task (`pending_tasks == 0`); `thread_pool_wait` blocks on it.
- `thread_pool_submit` returns `false` (rejecting the task) if the queue is full or the pool is shutting down. Callers are expected to check this; the current `dot_product` submits at most `num_threads` tasks per call, so with the default capacity of 1024 the queue never fills.
- `thread_pool_destroy` sets `shutdown`, broadcasts `notify` so every worker wakes, joins all threads, then destroys the primitives and frees memory.

The worker loop releases the lock while executing a task, so multiple workers run concurrently and only contend when picking up or finishing work.

### Parallel `dot_product`

`dot_product(A, B, pool)` computes `C = A·B` by splitting the **rows of A** across up to `num_threads` tasks (fewer if `A` has fewer rows than threads). Each `DotTask` records its `[start_row, end_row)` slice and writes only its own rows of `C`, so no synchronisation is needed inside the workers. The main thread submits every slice, calls `thread_pool_wait`, and returns `C`.

In the default model the largest multiplications per batch are `W₁ (64×7) · X (7×1024)` in the forward pass and `dZ₁ (64×1024) · A₀ᵀ (1024×7)` in the backward pass. Both are split over 64 rows, so 12 workers get 5 or 6 rows each. The output layer's `1×32` weight matrix produces a single task.

Passing `pool == NULL` runs a plain triple loop on the calling thread, useful for debugging.

## 7. Memory model of a training step

The design goal recorded in [mini_batching_results.md](../mini_batching_results.md) is "zero-allocation mini-batching" at the batch level:

- `train_network` allocates `batch_input` (F × B) and `batch_expected` (T × B) **once** before the epoch loop and reuses them for every batch of every epoch, and again for validation.
- For the last, ragged batch (`train_samples % B` samples) the code temporarily shrinks the matrices' `cols` field to the real batch size. The underlying `data` buffer is untouched, so no reallocation happens, and every downstream operation (`dot_product`, `mse_derivative`, `get_batch`) sees the correct width. The `cols` are restored to `B` afterwards.
- Within the forward and backward passes the matrix library still allocates temporaries (caches, transposes, gradients). These are freed each batch: `layer_backward` frees its temporaries as it goes, and `network_free_caches` releases the three per-layer caches after every batch (and after every validation/test batch too, since `layer_forward` always caches).
- Gradients (`d_weights`, `d_biases`) are allocated at layer creation, replaced with fresh matrices by `layer_backward` (old ones freed), and finally released by `free_layer`.
- Adam's moment matrices are allocated once in `create_adam_optimizer` and freed in `free_optimizer`.

Ownership summary: anything returned by a `matrix_*` function or `network_forward` belongs to the caller; everything stored inside a `layer`, `network`, `Dataset` or `Optimizer` is freed by that object's `free_*` function.

## 8. Benchmark logger

`BenchLogger` wraps a `FILE*` plus a `struct timespec`. `logger_start_epoch` reads `CLOCK_MONOTONIC`; `logger_log_epoch` reads it again, converts the difference to milliseconds, accumulates a running total, derives RMSE from the MSE it is handed, and appends one CSV row (`epoch,epoch_time_ms,total_time_ms,rmse_ms,mae_sec`) with an `fflush` so the file is valid even if the run is interrupted. It also prints the same information to stdout.

Because `logger_log_epoch` is called after validation, `epoch_time_ms` includes both the training batches and the validation pass.

## 9. The training loop in `main.c`

`train_network(nn, data, epochs, batch_size, opt, loss_deriv_func, pool, logger)`:

1. Allocate the two batch buffers.
2. For each epoch: start the timer, zero the accumulators.
3. For each window of `batch_size` training samples: adjust for a ragged tail, `get_batch`, `network_forward`, accumulate de-normalised squared and absolute errors, `network_backward` with the supplied loss derivative, `network_update_weights`, `network_free_caches`, free the prediction.
4. Reset the buffer widths, then run the same loop over the validation split **without** the backward pass or weight update.
5. Divide the accumulators by the sample counts, print train/val MAE, and hand train MSE/MAE to the logger.
6. Free the batch buffers.

`test_network(nn, dataset, batch_size, pool)` is the validation loop applied to the test split, printing MSE and MAE at the end.

Data order matters: the split is sequential and batches are taken in file order every epoch. There is no shuffling between epochs (see [FEATURES.md](FEATURES.md#known-limitations-and-quirks)).
