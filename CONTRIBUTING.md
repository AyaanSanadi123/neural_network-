# Contributing

Thanks for your interest in the project. This page explains how the code is organised for extension, the conventions to follow, and the recipes for the most common changes.

Before you start, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the data flow and [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for ownership rules. Almost every bug in a project like this is a memory-ownership mistake.

## Development setup

```bash
git clone https://github.com/AyaanSanadi123/neural_network-.git
cd neural_network-
```

Build and run exactly as described in [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md). There is no Makefile yet, so keep the full `gcc` command handy. Adding `-Wall -Wextra -g` during development is recommended.

For memory checks on Linux, `valgrind --leak-check=full ./neural_network` works out of the box (reduce `epochs` first). On macOS, build with `-fsanitize=address` instead.

## Code conventions (as used in the codebase)

- **One module per folder**, `name.h` + `name.c`, with a `#ifndef NAME_H` include guard. Headers are included by bare name and each folder is added with `-I`.
- **C99 with POSIX threads.** No C++ features, no compiler-specific extensions.
- **Structs are plain and public.** Fields are accessed directly; there are no getters.
- **`create_*` / `free_*` pairs** own allocation. Anything with a `create_` function has a matching `free_` function that releases everything inside it.
- **Matrix functions return new matrices.** If you add an operation, follow the same pattern unless you are deliberately adding an in-place variant (name it clearly, for example `matrix_add_inplace`).
- **Shape assumptions are asserted** with `assert` at the top of matrix operations.
- **Comments explain intent**, often in a conversational tone. Keep them accurate when you change behaviour.
- Indentation is 4 spaces. Pointer spacing is inconsistent in the existing code (`Matrix * m`, `Matrix* m`); prefer `Matrix* m` in new code.

## Recipes

### Add an activation function

1. In `activation_functions/activations.c`, implement `double myact(double x)` and `double myact_derivative(double x)`. The derivative must be expressed in terms of the pre-activation `x` (that is what `layer_backward` passes in).
2. Declare both in `activation_functions/activations.h`.
3. Use them: `create_layer(in, out, myact, myact_derivative)`.

No other file changes. `matrix_map` applies the function element-wise.

### Add a loss function

1. In `loss_functions/losses.c`, implement `Matrix* myloss_derivative(Matrix* predictions, Matrix* expected)` returning `∂L/∂Ŷ` with the same shape as `predictions`, averaged over the batch (divide by `predictions->cols`).
2. Declare it in `loss_functions/losses.h`.
3. Pass it to `train_network(..., myloss_derivative, ...)`.

If your loss needs a forward value for reporting (the current code computes MSE/MAE directly in `main.c`), add a `double myloss(Matrix*, Matrix*)` alongside and call it in the training loop.

### Add an optimizer

1. In `optimizer/optimizer.c`, write `void myopt_update(Optimizer* opt, layer* l, int layer_index)` that reads `l->d_weights` / `l->d_biases` and updates `l->weights` / `l->biases`.
2. If it needs per-layer state (momentum buffers, etc.), define a state struct in `optimizer.h`, allocate it in a `create_myopt_optimizer(...)` that receives `num_layers` and `layers` so it can size the buffers, and store it in `opt->state`.
3. Extend `free_optimizer` to release your state. Note that `network_update_weights` assumes any non-NULL `state` is an `AdamState` when it increments `t`; if your optimizer has state but no timestep, either give it a leading `int t` field or generalise that check.
4. Declare the constructor in `optimizer.h`.

### Change the network architecture

Edit `main()`:

```c
network* nn = create_network(4);
nn->layers[0] = create_layer(7,  128, relu,   relu_derivative);
nn->layers[1] = create_layer(128, 64, relu,   relu_derivative);
nn->layers[2] = create_layer(64,  16, relu,   relu_derivative);
nn->layers[3] = create_layer(16,   1, linear, linear_derivative);
```

Rules: the count passed to `create_network` equals the number of assigned slots; each layer's input size equals the previous layer's output size; the first input size equals `num_features`; the last output size equals `target_features`. Create the optimizer **after** the layers, because Adam sizes its caches from them.

### Add a matrix operation

1. Implement it in `matrix/matrix.c`, asserting shapes first and returning a freshly created matrix.
2. Declare it in `matrix/matrix.h`.
3. If it is expensive and row-separable, consider parallelising it the way `dot_product` does: define a task struct, a `static void worker(void*)`, split the rows, `thread_pool_submit` each slice, `thread_pool_wait`, free the task array.

### Add a dataset feature

See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md#adapting-the-pipeline). Remember to keep the target as the **last** column and to update `create_dataset(rows, N, 1)` and the first layer's input size.

## Checklist before opening a pull request

- [ ] The full build command compiles with no warnings under `-Wall -Wextra`.
- [ ] A short run (for example `epochs = 3`) completes, prints sensible MAE values, and exits with `--- SHUTDOWN SUCCESSFUL ---`.
- [ ] Every `create_*` you added has a `free_*`, and every matrix you allocate in a loop is freed in that loop.
- [ ] If you changed the CSV loader or normalisation, `training_benchmark.csv` still has the same columns (or the docs are updated).
- [ ] Docs updated: [docs/API_REFERENCE.md](docs/API_REFERENCE.md) for new public functions, [docs/FEATURES.md](docs/FEATURES.md) for new capabilities or removed limitations.
- [ ] Do not commit `training_benchmark.csv` changes from experiments, rebuilt binaries, or files from `myenv/`.
- [ ] Write a commit message that says what changed (the history has many empty `;` messages; let us not add more).

## Reporting problems

Open a GitHub issue with the exact build command, compiler version, operating system, the console output up to the failure, and (if relevant) the first few lines of the CSV you were loading.
