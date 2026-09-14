# Documentation Index

| Document | Contents |
| --- | --- |
| [GETTING_STARTED.md](GETTING_STARTED.md) | Prerequisites, build commands for macOS/Linux/Windows, running, reading the output, changing the experiment, training on your own CSV, troubleshooting. |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Annotated tree of the repository and what every folder and file is for, plus the module dependency graph. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the engine works: matrix layout, forward and backward pass maths, SGD and Adam, the thread pool, the memory model, and the training loop. |
| [API_REFERENCE.md](API_REFERENCE.md) | Every public struct and function, module by module, with ownership rules. |
| [DATA_PIPELINE.md](DATA_PIPELINE.md) | Where the F1 dataset comes from, what the cleaning scripts do, the final schema, and how to rebuild or extend it. |
| [FEATURES.md](FEATURES.md) | What is implemented, reference results, known limitations and quirks, and a roadmap. |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Conventions and step-by-step recipes for adding activations, losses, optimizers, layers and features. |
| [../mini_batching_results.md](../mini_batching_results.md) | Benchmark write-up of the move from single-sample SGD to mini-batching. |

Suggested reading order for a newcomer: README at the repository root, then GETTING_STARTED, then ARCHITECTURE, and keep API_REFERENCE open while reading the code.
