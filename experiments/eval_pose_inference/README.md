# Hierarchical Search over the Compatibility Landscape

This experiment implements Section 4.4 of the ContinuLoc paper. Given a query image, its visual encoder, and a scene-specific neural field, it estimates a continuous 4DoF pose through global exploration, population contraction, and CMA local refinement.

## Evaluate the supplied models

Install the pose inference dependencies and build Kaolin Wisp using the [environment guide](../../requirements/README.md).

Download the data from [ContinuScenes on Hugging Face](https://huggingface.co/datasets/wk666/ContinuScenes). The example below uses scene 03, so you can download just the manifest and that scene. Install `huggingface_hub` for the `hf` command and run `hf auth login` if the repository requires authentication.

```bash
hf download wk666/ContinuScenes --repo-type dataset \
  --include 'dataset_manifest.json' '03/**' \
  --local-dir /path/to/ContinuScenes
```

Keep `dataset_manifest.json` and `03/` under the same ContinuScenes directory. The manifest resolves the UAV images, aligned satellite maps, pose CSV, and metadata within the downloaded layout.

Download the [published reproduction bundle](https://huggingface.co/wk666/ContinuLoc/tree/main/paper_reproduction_bundle) and the two checkpoints for VisLoc 03:

```bash
hf download wk666/ContinuLoc --repo-type model \
  --include 'paper_reproduction_bundle/**' \
    'visloc_interval_train80_mad/shared/visual_encoder_epoch021.pth' \
    'visloc_interval_train80_mad/visloc_03/neural_field_epoch999.pth' \
  --local-dir /path/to/ContinuLoc-models
```

Keep the downloaded model and dataset layouts. In `paper_reproduction_bundle/run.yaml`,
set `manifest_dir` to the downloaded ContinuScenes directory. The supplied
`model_dir: ..` resolves the checkpoints from the parent directory.

Run from the ContinuLoc code repository:

```bash
python experiments/eval_pose_inference/run.py \
  --config /path/to/ContinuLoc-models/paper_reproduction_bundle/run.yaml --dry-run
```

Remove `--dry-run` to run inference. Set `eval_n_samples: 1` for a single-query
check or `null` for full evaluation. Adjust `hardware` and `output_dir` in
`run.yaml` as needed.

The default evaluates VisLoc 03 using the interval split and MAD scale filtering.
For another scene, download its data and weights and select the matching experiment
and scene in `run.yaml`. The bundle supplies the frozen evaluation split, model
settings, and thresholds.

| Experiment | Scenes |
|---|---|
| `visloc_interval_train80_mad` | `visloc_03`, `visloc_04` |
| `wingtra_interval_train90_mad` | `zurich`, `zuchwil` |

## Evaluate your own trained field

Use `pose_inference_batch.yaml` for a new single-scene model. Fill `load_stage2_ckpt` and its matching `inherit_stage2_yaml`, then set `enabled: true`:

```bash
python experiments/eval_pose_inference/run_batch.py --dry-run
python experiments/eval_pose_inference/run_batch.py
```

The example inherits `configs/pose_inference.yaml`. Set its `scenes_setting.manifest_dir` to your data directory. Leave `selected_scene_name` empty to inherit the field's scene; an explicit selection must match the model. Visual encoder weights and settings are recovered from the field's training configuration.

For a single complete configuration:

```bash
python experiments/eval_pose_inference/run.py --config /path/to/inference.yaml
```

Batch entries execute sequentially and stop on failure. Use `--base-yaml` and `--batch-config` to select other configurations. `--python` selects the inference interpreter. The supplied paper bundle is a separate input format consumed by `run.py`, not by `run_batch.py`.

## Search and evaluation settings

- `hds_stage1_*`: global candidate search, including rotation and scale sampling.
- `hds_stage2_*`: population contraction and candidate survival.
- `hds_stage3_*`: local CMA refinement.
- `hds_stage1_gallery_feature_source`: descriptors from the neural field (`ingp`) or visual encoder (`visual_encoder`) for initial search.
- `hds_scale_select_scope`: per-anchor scale estimation or query-level `global_pose_max` selection.
- `hds_eval_n_samples`: query limit; `null` evaluates all available queries.

These stage numbers refer to the three search steps, not the encoder/field/inference training workflow. Use the bundle's complete settings for its pretrained models. Evaluation thresholds are supplied separately from search parameters and are recorded with each run.

## Results

Each result directory contains progressive recall and error statistics (`stage3_progressive_recall_report.json`), poses and scores (`predictions.pt`), timing (`timing.json`), configuration snapshots, `manifest.json`, and `console.log`.

Bundle runs write beneath the entry's `output_dir`. Other runs default to `gen_fm_exps/inference_stage3/<run_id>/`. Logs are also saved under `gen_fm_exps/logs/inference_stage3/<run_id>/`. Run identifiers include a timestamp.

Reports distinguish validation and test members when present. Compare results using matching splits, query counts, and evaluation thresholds; a limited-query check is not a full-test result. Coordinate error units depend on the scene's coordinate system.
