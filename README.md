# ContinuLoc
<p align="center">
  <img src="./ContinuLoc.png" alt="ContinuLoc">
</p>

Continuous Pose Inference over Neural Fields for UAV Geo-Localization.

This `continuloc_v2_shared` branch retains the `train_stage1`, `train_stage2`, and
`inference_stage3` experiments. See [publication notes](docs/public_release.md)
for external dependencies, configuration portability, and Git history scope.

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

Edit the YAML files under `configs/` to point to your dataset, checkpoint, and output locations before running experiments.

## Landscape-Aware Representation Learning

Train the 4DoF-aware observation representation:

```bash
python experiments/train_stage1/run.py \
  --p_yaml configs/stage1_visual_encoder_wingtra.yaml
```

VisLoc scenes are resolved from `scenes_setting.dataset_manifest`. Stage 1 defaults
to scenes 03 and 04; Stage 2 defaults to scene 03. Select scenes by numeric ID or
full name (`03` / `visloc_03`), without copying dataset paths into the training YAML:

```bash
python experiments/train_stage1/run.py \
  --p_yaml configs/stage1_visual_encoder_visloc.yaml \
  --selected_scene_names 03 04 08 11
```

For one scene, use `--selected_scene_name 08` instead. These options override the
YAML selection and cannot be combined. Omitting both selection fields in a
manifest-based YAML uses the manifest's `default_scene_names`. Relative dataset
paths resolve against the manifest directory; keep the manifest path absolute.
Unknown, duplicate, unprepared scenes and missing paths fail before data loading.
Expanded scene names and paths are included in saved experiment configurations.
CSV quality flags and scale filtering remain separate from split generation. Stage 2
checkpoint settings still need to match the intended experiment.

Evaluate the learned representation with a reference gallery:

```bash
python experiments/train_stage1/run.py \
  --p_yaml configs/stage1_visual_encoder_wingtra.yaml \
  --test_only true
```

## Semantic Neural Field as a Continuous Pose Index

Train the semantic neural field to regress descriptors from 4DoF pose coordinates:

```bash
python experiments/train_stage2/run.py \
  --p_yaml experiments/train_stage2/config_wingtra.yaml
```

Set `load_stage1_ckpt` to Stage 1 weights and `inherit_stage1_yaml` to its `opts.yaml`
in the experiment config first. See [Stage 2 instructions](experiments/train_stage2/README.md).

Evaluate the learned field (also set `load2test` to the Stage 2 checkpoint):

```bash
python experiments/train_stage2/run.py \
  --p_yaml experiments/train_stage2/config_wingtra.yaml \
  --test_only true
```

## Hierarchical Pose Inference in Continuous Pose Space

Run continuous pose inference with the trained representation and semantic field:

```bash
python scripts/localize_stage3.py \
  --stage3_yaml configs/stage3_ingp_minimal.yaml \
  --stage2_opts_yaml configs/stage2_INGP_wingtra.yaml
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

- `configs/stage1_visual_encoder_wingtra.yaml`
- `configs/stage1_visual_encoder_visloc.yaml`
- `configs/stage2_INGP_wingtra.yaml`
- `configs/stage2_INGP_visloc.yaml`
- `configs/stage3_ingp_minimal.yaml`
- `configs/stage3_recall_thresholds.yaml`

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
python -m continuloc.datasets.dataset_neuloc_4d_uav_sat_pair \
  --scenes 08 11 --samples 8 --augmentation --query-geometry-method grid_sampling \
  --compare-geometry-dir /tmp/query_geometry_comparison \
  --visualize-dir /tmp/query_geometry_pairs \
  --report /tmp/query_geometry_comparison/report.json
python -m unittest discover -s tests -v
```

The comparison timing includes image decoding and positive satellite sampling,
uses `num_workers=0`, and excludes scene initialization and plots. It is a small
CPU smoke benchmark, not a measurement of full training throughput or accuracy.

## Repository layout

公共实现位于 `continuloc/`，启动命令位于 `scripts/`，配置位于 `configs/`。
仅用户明确指定保留的实验放在 `experiments/`；其余实验、诊断和过程脚本默认放在 `tmep/`。
详见 [组织约定](docs/project_layout.md) 和 [诊断入口](experiments/README.md)。

## Stage1 loss selection

在 Stage1 YAML 的 `learning_setting` 中使用 `loss_type: triplet_single_edge`
或 `loss_type: infonce`。两份配置默认使用 `triplet_alpha: 5.0`、softplus、
关闭行过滤、关闭可学习 alpha，与 master 基线配置对齐。
InfoNCE 使用 master 的温度缩放公式，类名为 `InfoNCE`。
可学习 loss 参数会加入优化器并保存于检查点的 `loss_fn` 字段。
旧 `triplet_loss` 和历史 mask-triplet 名称映射到新实现，公式不再是旧版未缩放 loss。

## Train / validation / test

```yaml
data_setting:
  split_source: generated
  split_mode: segment # segment / interval / random
  split_train_ratio: 0.7
  split_val_ratio: 0.1
  # test = 1 - train - val = 0.2
```

两份 Stage1 配置的验证比例默认是 0.0。比例针对有效性及边缘过滤后的样本；
先生成 train 与剩余样本，再从剩余样本中按总样本数确定 val 数量，余下为 test。
修改 train 或 val 比例会重新生成划分；仅改变 val 时 train 不变。
随机划分使用固定种子，同一数据与配置重复运行得到相同成员。
样本数取整会使实际比例略有偏差；正验证比例若导致 val/test 为空则明确报错。

`generated` 忽略历史 split 列（包括旧列的空成员），不修改 CSV。
以后使用固定 CSV 协议时显式设 `split_source: csv`；仍通过现有
`split_<mode><ratio>` 列名选列，缺列报错。支持 train/val/test（validation 视为 val）；
显式 val 成员优先于比例。CSV 未提供 val 时可以从其 test 中按比例划出，
不增加 CSV 未分配成员。独立 JSON 划分协议尚未实现。

Stage1/2 的 val 与 test 合并读取、共用一次图库构建，每个 batch 仅计算一次
匹配矩阵，再分别输出 `[scene][val]` / `[scene][test]` Recall 和样本数。
无 val 时保留原来的扁平指标返回结构；有 val 时返回 val/test 两个指标字典。
Stage3 同样执行一次合并查询推理后分别统计；设置样本数上限时仅报告实际评估成员。
Stage2 可在 data_setting 中覆盖这两个比例，也可继承 Stage1 保存的设置。

全局随机负样本按 batch 生成：

```yaml
data_setting:
  n_global_negatives_per_batch: 384  # 独立于 query batch 大小；0 表示关闭
  reject_sampling: false           # 开启后避开 batch 中全部 query 的邻域
```

`UAVSatPairDataset` 只构造 query/正样本对，`UAVSatBatchCollator` 在组批后
一次生成共享负样本池，输出 `satimgs_neg[N,C,H,W]`、`coords_sat_neg[N,4]`。
`reject_sampling` 同时考虑可选的卫星 query。旋转、尺度继续随机采样。
不再使用 `n_neg_per_query`、`add_random_satimg_negs`、`reject_batch_aware`；
Stage 1 不再通过 `batchsize_sat / batchsize_uav` 推导负样本数。
Stage 2/3 的 `batchsize_sat` 仍保留其原有用途。

场景 manifest 中配置 `global_negative_sampling_grid` 时，从其中的有效
100 m 网格采样。边界裁剪后的格子按剩余面积加权，再在格内随机取点；
无网格场景从现有采样边界内均匀采样。网格仅作用于额外的全局负样本，
不改变正样本 CSV、数据划分、卫星 query 或评估图库。距离排除与网格约束
独立，补采失败明确报错，不回退到不受约束的随机坐标。
运行日志记录网格 NPZ 的 SHA256 和裁剪后的可采样区域数量。
