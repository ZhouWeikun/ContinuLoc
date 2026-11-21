# Neural Compression of Pose-conditioned Semantic Fields

This experiment implements Section 4.3 of the ContinuLoc paper. It compresses descriptors from the trained visual encoder into a scene-specific semantic neural field. The visual encoder remains frozen while the hash grid and coordinate-conditioned decoder learn to predict descriptors from pose coordinates.

Each run trains one scene. The supervising visual encoder may have been trained on multiple scenes. For evaluation with supplied pretrained fields, proceed directly to the [reproduction workflow](../../README.md#reproduce-inference-with-the-supplied-models).

## Train

Install the neural field dependencies and build Kaolin Wisp using the [environment guide](../../requirements/README.md).

Set `scenes_setting.manifest_dir` in `configs/neural_field_continuscenes.yaml` to your ContinuScenes metadata directory. Provide the visual encoder checkpoint and the `opts.yaml` saved with it:

```bash
python experiments/train_neural_field/run.py \
  --p_yaml configs/neural_field_continuscenes.yaml \
  --selected_scene_name visloc_03 \
  --load_stage1_ckpt /path/to/visual_encoder/model.pth \
  --inherit_stage1_yaml /path/to/visual_encoder/opts.yaml \
  --exp_name_override neural_field_example
```

Select any prepared scene by name. The equivalent public command is `python -m continuloc.cli.train_neural_field`.

The default supervision uses satellite crops only (`data_setting.use_uavimgs: false`). Set this to `true` to also use UAV images in training. UAV validation/test queries remain available in either mode.

Explicit settings in the neural-field YAML take precedence over inherited settings. Match the visual encoder's data filtering and split protocol when configuring training.

## Model structure and checkpoints

The shared hash-grid configuration is `configs/neural_field_hash_grid_continuscenes.yaml`. Its default `codebook_bitwidth` is 19; Zurich and VisLoc experiments have used both 18 and 19. When restoring a model, use its actual training structure, including hash-grid width, decoder dimensions, and positional encoding.

Add `--load2train /path/to/neural_field/model.pth` to resume training. To evaluate a trained field, use its matching configuration with `--test_only true --load2test /path/to/neural_field/model.pth`. The field checkpoint requires its supervising visual encoder and configuration as well.

Logs and checkpoints are saved under the configured `dir2save_log` and `dir2save_ckpt`. Preserve the weights and generated `opts.yaml` for subsequent pose inference.

## Batch example

`neural_field_batch.yaml` contains one disabled example. Set the scene and the two visual encoder paths, then set `enabled: true`.

```bash
python experiments/train_neural_field/run_batch.py --dry-run
python experiments/train_neural_field/run_batch.py
```

The preview generates resolved YAML and commands without loading models, validating checkpoint contents, or training. Each entry inherits `configs/neural_field_continuscenes.yaml`; copy the entry with a new name for another scene or experiment.

Use `--base-yaml`, `--batch-config`, and `--output-dir` to choose configuration files and the generated-config directory. `--python /path/to/environment/bin/python` selects the training interpreter. Use an environment with the required Kaolin/Wisp CUDA extensions.

Satellite scale sampling can be configured through `sat_scale_sampling_mode` (`empirical`, `uniform_linear`, or `uniform_log`), `sat_scale_jitter_enabled`, and `sat_scale_jitter_half_span_ratio`. These are experiment choices; use the supplied model bundle's settings for paper evaluation.
