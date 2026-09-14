# API Reference

Public functions and structs, grouped by module. Unless stated otherwise, any `Matrix*` returned by a function is newly allocated and owned by the caller, who must release it with `free_matrix`.

Include paths: every module folder must be on the include path (`-I<folder>`), and headers are included by bare name (`#include "matrix.h"`).

---

## `matrix/matrix.h`

```c
typedef struct {
    int cols;
    int rows;
    double* data;   // row-major, rows*cols entries; (i,j) at data[i*cols + j]
} Matrix;
```

| Function | Description |
| --- | --- |
| `Matrix* create_matrix(int rows, int cols)` | Allocate a zero-filled matrix. |
| `void free_matrix(Matrix* m)` | Free the data buffer and the struct. Safe on `NULL`. |
| `Matrix* dot_product(Matrix* a, Matrix* b, ThreadPool* pool)` | `a · b`. Asserts `a->cols == b->rows`. If `pool` is non-NULL the rows of `a` are split across up to `pool->num_threads` tasks and the call blocks until all finish; if `NULL` it runs on the calling thread. |
| `Matrix* transpose(Matrix* m)` | Returns `mᵀ`. |
| `Matrix* hadamard_product(Matrix* a, Matrix* b)` | Element-wise product. Asserts equal shapes. |
| `Matrix* matrix_copy(Matrix* m)` | Deep copy. |
| `Matrix* matrix_add(Matrix* a, Matrix* b)` | Element-wise sum. Asserts equal shapes. |
| `Matrix* matrix_subtract(Matrix* a, Matrix* b)` | Element-wise difference. Asserts equal shapes. |
| `Matrix* matrix_multiply_scalar(Matrix* m, double scalar)` | Scales every element. |
| `Matrix* matrix_map(Matrix* m, double (*func)(double))` | Applies `func` to every element (used for activations and their derivatives). |
| `Matrix* matrix_add_bias(Matrix* wx, Matrix* bias)` | Adds the `(rows × 1)` column vector `bias` to every column of `wx`. Asserts `bias->cols == 1` and matching rows. |
| `Matrix* matrix_sum_columns(Matrix* m)` | Returns a `(rows × 1)` matrix whose entry `i` is the sum of row `i`. Used to reduce `dZ` into the bias gradient. |

---

## `layers/layer.h`

```c
typedef struct {
    int input_size;
    int output_size;
    Matrix* weights;            // (output_size × input_size)
    Matrix* biases;             // (output_size × 1)
    Matrix* input_cache;        // A_prev from the last forward pass, or NULL
    Matrix* z_cache;            // Z = W·A_prev + b, or NULL
    Matrix* activation_cache;   // A = g(Z), or NULL
    Matrix* d_weights;          // ∂L/∂W from the last backward pass
    Matrix* d_biases;           // ∂L/∂b from the last backward pass
    double (*activation_func)(double);
    double (*activation_derivative)(double);
} layer;
```

| Function | Description |
| --- | --- |
| `double random_uniform()` | Returns a value in [-0.5, 0.5) from `rand()`. No seeding is performed anywhere, so initial weights are identical on every run. |
| `layer* create_layer(int input_size, int output_size, double (*act_func)(double), double (*act_deriv)(double))` | Allocate a layer with uniformly random weights, zero biases, NULL caches and zero gradient buffers. |
| `void layer_free_caches(layer* l)` | Free and NULL the three forward caches. Safe to call when they are already NULL. |
| `void free_layer(layer* l)` | Free weights, biases, gradients, caches and the struct. |

---

## `activation_functions/activations.h`

All functions are `double → double` and are meant to be passed to `create_layer` in matching pairs.

| Function | Definition |
| --- | --- |
| `relu(x)` | `x` if `x > 0`, else `0` |
| `relu_derivative(x)` | `1` if `x > 0`, else `0` |
| `sigmoid(x)` | `1 / (1 + e^-x)`, returning exactly `1.0` for `x > 6` and `0.0` for `x < -6` |
| `sigmoid_derivative(x)` | `sigmoid(x) · (1 - sigmoid(x))` |
| `linear(x)` | `x` |
| `linear_derivative(x)` | `1` |

---

## `loss_functions/losses.h`

| Function | Description |
| --- | --- |
| `Matrix* mse_derivative(Matrix* predictions, Matrix* expected)` | Returns `(predictions - expected) / predictions->cols` element-wise, i.e. the gradient of half the mean squared error with respect to the output, averaged over the batch. Asserts equal shapes. Matches the `loss_deriv_func` signature expected by `network_backward` and `train_network`. |

---

## `network/fp.h`

```c
typedef struct {
    int num_layers;
    layer** layers;   // array of num_layers layer pointers, filled in by the caller
} network;
```

| Function | Description |
| --- | --- |
| `network* create_network(int num_layers)` | Allocate the container and an uninitialised `layers` array. The caller must assign every slot with `create_layer` before use. |
| `Matrix* layer_forward(layer* l, Matrix* input, ThreadPool* pool)` | Compute and cache `A = g(W·input + b)`. Returns the layer's own `activation_cache` (not a copy; do not free it). |
| `Matrix* network_forward(network* nn, Matrix* network_input, ThreadPool* pool)` | Run every layer in order. Returns a **copy** of the final activation, owned by the caller. Leaves caches populated for a subsequent backward pass. |
| `Matrix* layer_backward(layer* l, Matrix* dA, ThreadPool* pool)` | Given `∂L/∂A` for this layer, compute and store `d_weights` and `d_biases`, and return `∂L/∂A_prev` (caller-owned). Requires the caches from a preceding `layer_forward`. |
| `void network_backward(network* nn, Matrix* predictions, Matrix* expected, Matrix* (*loss_deriv_func)(Matrix*, Matrix*), ThreadPool* pool)` | Compute the loss gradient with `loss_deriv_func` and backpropagate through all layers, filling every layer's gradients. |
| `void network_update_weights(network* nn, Optimizer* opt)` | Increment the Adam timestep if the optimizer has state, then call `opt->update_func` on every layer. |
| `void network_free_caches(network* nn)` | Call `layer_free_caches` on every layer. Call this after each batch (including forward-only validation/test batches). |
| `void free_network(network* nn)` | Free every layer, the layer array and the struct. Does not free the optimizer. |

---

## `optimizer/optimizer.h`

```c
typedef struct { Matrix *m_weights, *v_weights, *m_biases, *v_biases; } AdamLayerCache;

typedef struct {
    int t;                        // timestep, incremented per network_update_weights
    double beta1, beta2, epsilon; // 0.9, 0.999, 1e-8
    AdamLayerCache* layer_caches; // one per layer
} AdamState;

struct Optimizer {
    double learning_rate;
    void*  state;                 // AdamState* for Adam, NULL for SGD
    void (*update_func)(Optimizer* opt, layer* l, int layer_index);
};
```

| Function | Description |
| --- | --- |
| `Optimizer* create_sgd_optimizer(double learning_rate)` | Stateless gradient descent: `w -= lr·dw`, `b -= lr·db`. |
| `Optimizer* create_adam_optimizer(double learning_rate, int num_layers, layer** layers)` | Adam with zero-initialised moment matrices shaped from the given layers. Must be created **after** the layers. |
| `void adam_update(Optimizer* opt, layer* l, int layer_index)` | The Adam step for one layer; installed as `update_func` by `create_adam_optimizer`. |
| `void free_optimizer(Optimizer* opt, int num_layers)` | Free the state (if any) and the struct. `num_layers` must match what was passed at creation. |

`sgd_update` exists in `optimizer.c` but is not declared in the header; use `create_sgd_optimizer`.

---

## `thread_pool/threadpool.h`

```c
typedef struct { void (*execute)(void* arg); void* arg; } Task;

typedef struct {
    pthread_t* threads;  int num_threads;
    Task* task_queue;    int queue_capacity, head, tail, count, pending_tasks;
    pthread_mutex_t lock;
    pthread_cond_t notify;     // work available
    pthread_cond_t all_idle;   // pending_tasks reached 0
    bool shutdown;
} ThreadPool;
```

| Function | Description |
| --- | --- |
| `ThreadPool* thread_pool_init(int num_threads, int queue_capacity)` | Start the workers. Returns `NULL` (after cleaning up) on invalid arguments or any allocation/thread failure. |
| `bool thread_pool_submit(ThreadPool* pool, void (*execute)(void*), void* arg)` | Enqueue a task. Returns `false` if the queue is full or the pool is shutting down. The `arg` must stay valid until the task has run. |
| `void thread_pool_wait(ThreadPool* pool)` | Block until every submitted task has finished. |
| `void thread_pool_destroy(ThreadPool* pool)` | Signal shutdown, let workers drain the queue, join them, destroy the primitives and free the pool. |

---

## `benchmarking_engine/b_engine.h`

```c
typedef struct {
    FILE* fp;
    double cumulative_time_ms;
    struct timespec epoch_start;
} BenchLogger;
```

| Function | Description |
| --- | --- |
| `BenchLogger* logger_init(const char* filename)` | Open (truncate) the CSV, write the header `epoch,epoch_time_ms,total_time_ms,rmse_ms,mae_sec`. Exits the process if the file cannot be opened. |
| `void logger_start_epoch(BenchLogger* logger)` | Record the epoch start time (`CLOCK_MONOTONIC`). |
| `void logger_log_epoch(BenchLogger* logger, int epoch, double mse, double mae_sec)` | Compute the elapsed time, update the running total, write one CSV row (RMSE is `sqrt(mse)`), flush, and print a summary line. |
| `void logger_close(BenchLogger* logger)` | Close the file and free the logger. |

---

## `data_preprocessing/data_loader.h`

```c
typedef struct {
    int num_features;      // input columns
    int target_features;   // target columns
    int total_samples;
    Matrix** raw_inputs;   // total_samples × (num_features × 1)
    Matrix** raw_targets;  // total_samples × (target_features × 1)
    int train_samples; Matrix** train_inputs; Matrix** train_targets;   // views into raw_*
    int test_samples;  Matrix** test_inputs;  Matrix** test_targets;
    int val_samples;   Matrix** val_inputs;   Matrix** val_targets;
    double target_min, target_max;   // set by normalize_dataset, from the training split
} Dataset;
```

| Function | Description |
| --- | --- |
| `void count_csv_dimensions(const char* filepath, int* out_rows, int* out_cols, int has_header)` | Count data rows (excluding the header if `has_header`) and columns (commas in the first line + 1). Exits on open failure. |
| `Dataset* create_dataset(int total_samples, int num_features, int target_features)` | Allocate the arena of per-sample matrices. Split views start as `NULL`. |
| `void load_csv(const char* filepath, Dataset* dataset, int has_header)` | Parse up to `total_samples` lines; the first `num_features` columns go to inputs, the rest to targets. Cells are parsed with `atof`; empty cells become `0.0`. Lines are limited to 4096 bytes. |
| `void split_dataset_sequential(Dataset* data, float train_ratio, float test_ratio, float val_ratio)` | Assign the first `train_ratio` of samples to train, the next `test_ratio` to test, and **all remaining** samples to validation (`val_ratio` is not used in the arithmetic). No shuffling. |
| `void normalize_dataset(Dataset* data)` | Min-max scale every input feature and the (single) target column using training-split statistics; stores `target_min`/`target_max`. Must be called after splitting. |
| `void get_batch(Dataset* data, Matrix** source_inputs, Matrix** source_targets, int start_index, int current_batch_size, Matrix* batch_input, Matrix* batch_expected)` | Copy samples `[start_index, start_index + current_batch_size)` from the given split into the preallocated batch matrices, one sample per column. Uses `batch_input->cols` as the row stride, so shrink `cols` for a ragged batch before calling. |
| `void free_dataset(Dataset* data)` | Free every sample matrix, the arena arrays, the split views and the struct. |

---

## `main.c`

Not a library module, but the two loop functions are reusable:

| Function | Description |
| --- | --- |
| `void train_network(network* nn, Dataset* data, int epochs, int batch_size, Optimizer* opt, Matrix* (*loss_deriv_func)(Matrix*, Matrix*), ThreadPool* pool, BenchLogger* logger)` | Mini-batch training over `data->train_*` with a validation pass over `data->val_*` after every epoch. Reports MAE in de-normalised target units and logs each epoch. |
| `void test_network(network* nn, Dataset* dataset, int batch_size, ThreadPool* pool)` | Forward-only evaluation over `dataset->test_*`; prints MSE and MAE. |
