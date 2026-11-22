# Configuration guide

ContinuLoc uses one ContinuScenes interface for all scenes. Choose scenes by name; data layout and coordinate metadata are resolved through the dataset manifest.

## Supplied-model evaluation

Use the configurations packaged with the pretrained models through the [paper reproduction entrypoint](../README.md#reproduce-inference-with-the-supplied-models). The bundle preserves model structure, search settings, thresholds, and evaluation membership. Its `configs/` files are templates resolved by `bundle.yaml`, together with `assets/`, `splits/`, and `contracts/`.

The YAML files in this repository are starting points for new experiments. They do not replace the complete configuration supplied with a pretrained model.

## Shared examples

| File | Purpose |
|---|---|
| `visual_encoder_continuscenes.yaml` | Visual encoder training across selected scenes |
| `neural_field_continuscenes.yaml` | Neural-field training for one selected scene |
| `neural_field_hash_grid_continuscenes.yaml` | Hash-grid structure, with a 19-bit default |
| `pose_inference.yaml` | Continuous pose inference for a trained field |
| `pose_inference_recall_thresholds.yaml` | Named evaluation thresholds and scene mappings |

Zurich and VisLoc experiments have used 18-bit and 19-bit grids. Restore each checkpoint with its matching structure. The reproduction bundle supplies the correct grid for each model.

Each directory in `experiments/` has one disabled batch example. Entries inherit the relevant shared configuration and override only experiment-specific values. Use a unique `name`, fill required paths, and set `enabled: true` when ready.

## Dataset location and scene selection

Download [ContinuScenes on Hugging Face](https://huggingface.co/datasets/wk666/ContinuScenes) and set its local directory in your training or custom inference YAML:

```yaml
scenes_setting:
  manifest_dir: /path/to/ContinuScenes
```

Keep `dataset_manifest.json`, the scene directories, and any `splits/<scene>.json` files together as distributed. Standard training protocols use the dataset's split files. Supplied-model evaluation uses the reproduction bundle's frozen splits and sets `manifest_dir` directly in `run.yaml`.

For visual encoder training, choose a list with `--selected_scene_names visloc_03 zurich`. Neural-field training uses one scene, for example `--selected_scene_name zurich`. Inference must use the scene associated with the selected field checkpoint.

## Fixed training and evaluation protocols

ContinuLoc reads fixed JSON splits by default. The shared examples use:

```yaml
data_setting:
  split_source: json
  split_mode: segment
  split_val_ratio: 0
  enable_edge_filter: true
  scale_filter_mode: none
```

`split_mode` selects `segment` or `interval`. Training/test proportions are recorded in each scene's fixed JSON; changing `split_train_ratio` does not regenerate these assignments. The standard scene protocols use 80/20 for VisLoc scenes and 90/10 for Zurich/Zuchwil, with no validation split.

| `enable_edge_filter` | `scale_filter_mode` | Stored profile |
|---|---|---|
| `false` | `none` | quality |
| `true` | `none` | quality + edge |
| `true` | `mad` | quality + edge + scale |

Each profile has its own filtered members and assignments. Runtime switches select the stored profile, without recomputing filters or train/test membership. MAD with edge filtering disabled has no corresponding profile. Match data settings between a visual encoder and the field trained from it.

For supplied-model evaluation, the bundle's frozen splits take precedence over standard dataset splits. Preserve them to evaluate the intended query set.

Custom experiments can use `split_source: generated` to generate assignments with explicit train/validation proportions. Such experiments do not reproduce the supplied frozen protocol merely by using the same scene names.

## Paths and outputs

Training checkpoints and logs default to `gen_fm_exps/ckpts/` and `gen_fm_exps/logs/`. Relative values for `dir2save_ckpt` and `dir2save_log` resolve against the repository root.

For batch files, relative `base_yaml` paths also resolve against the repository root. For a reproduction bundle entry, relative paths resolve against the entry file's directory; resource paths in `bundle.yaml` resolve against that bundle file, and model paths resolve against `model_dir`.

## Manifest path rules

The released schema-v2 manifest resolves scene resources relative to the
downloaded ContinuScenes directory. To relocate the dataset, move the directory
as a unit and update `manifest_dir`. Preserve satellite source order, geometry,
and normalization statistics. Satellite rasters live in each scene's
`sattifs_multisource_aligned/` directory.

For a custom model whose saved configuration refers to a missing grid file,
set `network_setting.p_grid_config_yaml` to an existing matching grid YAML.
There is no automatic substitution of legacy dataset-specific configurations.
The paper reproduction bundle resolves its own grid paths.
