# Public branch scope and audit

`continuloc_v2_shared` starts from the committed `continuloc_v2` revision
`1a660d1`. Uncommitted and untracked development-worktree changes are not included.
Only `train_stage1`, `train_stage2`, and `inference_stage3` remain under
`experiments/`. Other experiments are recoverable from the development branch
and Git history. Shared implementation and existing regression tests are retained.

## Remaining publication work

- YAML files still contain author-machine dataset, checkpoint, and output paths.
  Batch configurations include historical run settings, including a Stage 3
  reference to a development-only `tmep/` YAML. Replace these before running.
- Public code also contains machine-specific paths: the DINOv3 fallback weights
  path, dataset debug visualization defaults, and standalone dataset examples.
  Maintenance scripts have local log/checkpoint defaults.
- `pyproject.toml` declares package metadata but not runtime dependencies.
  PyTorch, vision/geospatial dependencies, pretrained weights, Kaolin Wisp and
  CUDA extensions must be prepared separately. Full training and inference have
  not been validated in a clean external environment.
- `trainers/` and `trainer_depends/configs` are compatibility adapters, not
  additional experiments. `docs/layout_migration.json` is a historical migration
  record and may mention experiments or local checks excluded from this branch.
- This is a normal descendant branch, not a history-free export. Excluded files
  remain accessible in ancestor commits. This structural audit is not a full
  credential, licensing, or third-party redistribution audit.

Do not interpret the three-experiment directory layout as a claim that this branch
is already a fully portable or fully audited public release.

## Checks performed

- All 103 retained Python source files parsed successfully.
- Only the three requested experiment directories remain. AST inspection found
  no direct imports from `experiments`, `scripts`, `tmep`, or `temp` in `continuloc`.
- The compatibility configuration symlink resolves and `git diff --check` passes.
- All eight existing regression tests passed with the local `neuloc_wisp`
  environment: `python -m unittest discover -s tests -v`. The base Python
  environment lacks PyTorch and cannot run these tests.

These checks do not exercise full dataset loading, training, or CUDA inference.
