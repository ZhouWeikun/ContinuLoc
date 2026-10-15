# ContinuLoc
<p align="center">
  <img src="./ContinuLoc.png" alt="ContinuLoc">
</p>

Continuous Pose Inference over Neural Fields for UAV Geo-Localization.

ContinuLoc formulates cross-view UAV geo-localization as continuous 4DoF pose inference over an observation-scene fitness landscape. The system has three main components aligned with the paper:

- **Landscape-Aware Representation Learning** learns 4DoF pose-sensitive UAV and reference descriptors.
- **Semantic Neural Field as a Continuous Pose Index** distills pose-conditioned scene evidence into a compact neural field.
- **Hierarchical Pose Inference in Continuous Pose Space** searches the induced fitness landscape from global exploration to local refinement.

The ContinuScenes dataset release is in progress and will be made available soon.

## Setup

Install the project dependencies in your preferred Python environment. The implementation expects PyTorch and the dependencies used by the visual backbone, SALAD aggregator, and INGP modules.

If Kaolin Wisp is installed outside this repository, expose it with:

```bash
export KAOLIN_WISP_ROOT=/path/to/kaolin-wisp
```

Edit the YAML files under `trainer_depends/configs/` to point to your dataset, checkpoint, and output locations before running experiments.

## Landscape-Aware Representation Learning

Train the 4DoF-aware observation representation:

```bash
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml trainer_depends/configs/stage1_visual_encoder_wingtra.yaml
```

VisLoc scenes are resolved from `scenes_setting.dataset_manifest`. Stage 1 defaults
to scenes 03 and 04; Stage 2 defaults to scene 03. Select scenes by numeric ID or
full name (`03` / `visloc_03`), without copying dataset paths into the training YAML:

```bash
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml trainer_depends/configs/stage1_visual_encoder_visloc.yaml \
  --selected_scene_names 03 04 08 11
```

For one scene, use `--selected_scene_name 08` instead. These options override the
YAML selection and cannot be combined. Omitting both selection fields in a
manifest-based YAML uses the manifest's `default_scene_names`. Relative dataset
paths resolve against the manifest directory; keep the manifest path absolute.
Unknown, duplicate, unprepared scenes and missing paths fail before data loading.
Expanded scene names and paths are included in saved experiment configurations.
CSV validity, split labels, and scale filtering rules are unchanged. Stage 2
checkpoint settings still need to match the intended experiment.

Evaluate the learned representation with a reference gallery:

```bash
python trainers/stage1_visual_encoder_minimal.py \
  --p_yaml trainer_depends/configs/stage1_visual_encoder_wingtra.yaml \
  --test_only true
```

## Semantic Neural Field as a Continuous Pose Index

Train the semantic neural field to regress descriptors from 4DoF pose coordinates:

```bash
python trainers/stage2_INGP_minimal.py \
  --p_yaml trainer_depends/configs/stage2_INGP_wingtra.yaml
```

Evaluate the learned field:

```bash
python trainers/stage2_INGP_minimal.py \
  --p_yaml trainer_depends/configs/stage2_INGP_wingtra.yaml \
  --test_only true
```

## Hierarchical Pose Inference in Continuous Pose Space

Run continuous pose inference with the trained representation and semantic field:

```bash
python trainers/stage3_ingp_inferencer_minimal.py \
  --stage3_yaml trainer_depends/configs/stage3_ingp_minimal.yaml \
  --stage2_opts_yaml trainer_depends/configs/stage2_INGP_wingtra.yaml
```

The inference entrypoint reports progressive recall for global exploration, population contraction, and local refinement.

Example inference results:

<p align="center">
  <img src="./search_taizhou03.png" alt="ContinuLoc inference result on Taizhou" width="100%">
  <br>
  <em>Taizhou scene.</em>
</p>

<p align="center">
  <img src="./search_zurich.png" alt="ContinuLoc inference result on Zurich" width="100%">
  <br>
  <em>Zurich scene.</em>
</p>

## Configuration

Common configuration files:

- `trainer_depends/configs/stage1_visual_encoder_wingtra.yaml`
- `trainer_depends/configs/stage1_visual_encoder_visloc.yaml`
- `trainer_depends/configs/stage2_INGP_wingtra.yaml`
- `trainer_depends/configs/stage2_INGP_visloc.yaml`
- `trainer_depends/configs/stage3_ingp_minimal.yaml`
- `trainer_depends/configs/stage3_recall_thresholds.yaml`

Use the `wingtra` or `visloc` variants according to the scene split you want to run.

### Optional grid sampling query geometry

The default query augmentation remains `pipelined`. To opt into a centered source
crop followed by antialiased reduction and one combined rotation/scale/shear
sampling operation, add these fields to the training YAML:

```yaml
data_setting:
  query_geometry_method: grid_sampling
  query_prefilter_resolution_factor: 2.0
```

The grid_sampling pipeline outputs the same tensor shape and tracks the same rotation and
scale label convention. It retains colour/erase/normalization steps and leaves
the test transform unchanged. It currently supports centered square crops and
x-shear; extra affine translation/rotation/scale and random crops are rejected.
Here `grid_sampling` describes the coordinate mapping; sampling uses Pillow's
affine transform, not PyTorch's `grid_sample`.
Shear remains an unlabelled training perturbation. Combining operations can retain
content that the pipelined intermediate square canvases discarded; padding pixels
and integer-rounded pipelined scale crops are not expected to match exactly.

`query_prefilter_resolution_factor: 0` disables antialiased source reduction for
comparison; directly sampling a large source can alias. The normal grid_sampling path
uses Lanczos reduction to roughly twice the output resolution (adjusted for
scale/shear), followed by one bicubic affine sample, not repeated 224px warps.
This is an approximate antialiasing strategy, not a guarantee of alias-free output.

Compare fixed-parameter real DataLoader pairs (same rows, rotation, scale and
shear; no erasure/colour jitter) and CPU timing:

```bash
python -m trainer_depends.datasets.dataset_neuloc_4d_uav_sat_pair \
  --scenes 08 11 --samples 8 --augmentation --query-geometry-method grid_sampling \
  --compare-geometry-dir /tmp/query_geometry_comparison \
  --visualize-dir /tmp/query_geometry_pairs \
  --report /tmp/query_geometry_comparison/report.json
python -m unittest discover -s tests -v
```

The comparison timing includes image decoding and positive satellite sampling,
uses `num_workers=0`, and excludes scene initialization and plots. It is a small
CPU smoke benchmark, not a measurement of full training throughput or accuracy.
