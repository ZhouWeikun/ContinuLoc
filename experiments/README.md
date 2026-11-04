# Public experiments

`continuloc_v2_shared` keeps three experiment workflows:

- `train_stage1/`: visual encoder training and evaluation.
- `train_stage2/`: single-scene coordinate feature-field training and evaluation.
- `inference_stage3/`: hierarchical continuous pose inference, including batch runs.

Reusable implementations live in `continuloc/`; general command adapters live in
`scripts/`. Research-only experiments remain on `continuloc_v2`.
Write generated outputs to `gen_fm_exps/`, checkpoints to `gen_fm_exps/ckpts/`,
and logs to `gen_fm_exps/logs/`.

Before running, replace machine-specific data and checkpoint paths in the YAML
files and install the required training dependencies. See
[publication notes](../docs/public_release.md) for the current audit limits.
