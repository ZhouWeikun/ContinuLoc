# Continuloc V2 shared file set

The public branch retains three experiment directories:

- `experiments/train_stage1/`: representation training and evaluation.
- `experiments/train_stage2/`: semantic neural-field training and evaluation.
- `experiments/inference_stage3/`: continuous pose inference.

Shared code, configuration, command adapters, regression tests, and compatibility
entrypoints remain in `continuloc/`, `configs/`, `scripts/`, `tests/`, `trainers/`,
and `trainer_depends/`. These directories are not an assertion that every file is
needed by every workflow or that all dependencies are packaged.

See [README](README.md) for commands and
[publication notes](docs/public_release.md) for the audit scope and remaining work.
