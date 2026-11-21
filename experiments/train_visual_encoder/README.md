# Learning Pose-Sensitive Visual Representations

This experiment implements Section 4.2 of the ContinuLoc paper. The visual encoder learns descriptors for pose-aligned UAV and satellite observations. A single encoder can be trained on multiple ContinuScenes scenes and used to supervise scene-specific neural fields.

To evaluate the supplied pretrained models, use the [reproduction workflow](../../README.md#reproduce-inference-with-the-supplied-models); this training step is optional.

## Train

Install the visual encoder dependencies using the [environment guide](../../requirements/README.md).

Set `scenes_setting.manifest_dir` in `configs/visual_encoder_continuscenes.yaml` to your ContinuScenes metadata directory. Ensure the manifest points to your data. From the repository root:

```bash
python experiments/train_visual_encoder/run.py \
  --p_yaml configs/visual_encoder_continuscenes.yaml \
  --selected_scene_names visloc_03 zurich \
  --exp_name_override visual_encoder_example
```

Choose any prepared scenes from the manifest. Without a command-line scene selection, the shared example uses its configured scene list. The equivalent public entrypoint is `python -m continuloc.cli.train_visual_encoder`.

The default data protocol reads fixed JSON assignments. Use `split_mode: segment` or `interval` to select a protocol. `enable_edge_filter` and `scale_filter_mode` select its stored filtering profile. See the [configuration guide](../../configs/README.md).

## Evaluate the representation

Supply a checkpoint trained with the matching network configuration. Copy its `opts.yaml` to a local evaluation YAML and update the data paths for your machine:

```bash
python experiments/train_visual_encoder/run.py \
  --p_yaml /path/to/visual_encoder/evaluation.yaml \
  --test_only true \
  --load2test /path/to/visual_encoder/model.pth
```

This evaluates descriptor retrieval; continuous pose inference is described in the [pose inference guide](../eval_pose_inference/README.md).

## Batch example

`visual_encoder_batch.yaml` contains one disabled joint-training example. Edit its scene list and overrides, then set `enabled: true`.

```bash
python experiments/train_visual_encoder/run_batch.py --dry-run
python experiments/train_visual_encoder/run_batch.py
```

The preview generates resolved YAML and commands without loading models or training. Each entry independently inherits `configs/visual_encoder_continuscenes.yaml`. Use `--base-yaml` for another base configuration and `--batch-config` for another batch list. Overrides accept unique short keys or dotted configuration paths.

Logs and checkpoints are saved under the configured `dir2save_log` and `dir2save_ckpt`, with the experiment name identifying each run. Keep the saved `opts.yaml` with its weights: neural-field training needs both.
