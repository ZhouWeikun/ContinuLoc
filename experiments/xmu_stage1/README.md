# 厦大翔安：冻结 Stage1 的二维检索评价

使用 VisLoc Stage1 的视觉编码器与 SALAD 聚合器，严格加载已有权重，不训练。
输入 `/home/data/zwk/data_uavimgs_XianganXmu_512h_lineClassed/dataset_xmu_meta`，
全部 8,901 张无人机图像参与跨数据集评价，不沿用训练场景的划分或质量过滤。

先修改本目录 `config.yaml`，再运行：

```bash
/root/miniconda3/envs/neuloc_v0/bin/python -u experiments/xmu_stage1/evaluate_xy.py \
  --config experiments/xmu_stage1/config.yaml
```

配置可修改 checkpoint、data_root、output_root、crop_m、stride_m、batch_size、
max_queries、gpu_id、rotation_mode、rotation_step_deg、max_forward_variants、rot_retrieval_mode 和 heading_temperature；命令行同名参数（下划线改为短横线）优先于 YAML。
相对数据、权重与输出路径以项目根目录为基准。权重同目录需保留匹配的 opts.yaml。

此前 epoch024.pth 仅为已完成的首轮基线，尚未认定为最佳权重。
后续按验证集与测试集整体表现比较已有权重后，替换 checkpoint 再评价；
选权重时应记录使用的指标及跨场景/集合汇总方式。


- 卫星图：xiangan_xmu5km2_03res.tif，WGS84 地理坐标。
- 图库按元数据米分辨率采样约 100 m × 100 m 的朝北裁剪，中心间距约 25 m。
  只保留完整裁剪。100 m 是显式实验参数，并非由 UAV 真实覆盖尺度推导。
- UAV 中心方形裁剪后缩放到模型输入尺寸。默认 rotation_mode: search，
  对原始 query 独立应用 0,10,...,350 度共 36 个旋转，不使用真实朝向预先转正。
  由 rot_retrieval_mode 控制 Top10 排序，见下文；记录各候选方向。
  尺度固定，实际搜索 x/y/rotation 三个自由度；xy Recall 不附加方向命中条件。
  同时报告 Top1 方向误差（包括位置检索错误的样本）。元数据朝向仅用于事后评价。
- rotation_mode: metadata 复现旧的已知方向 2DoF 基线。
  旋转均复用公共训练器的评价变换；不归一尺度。
- 使用各自元数据 mean/std 归一化、完整 Stage1 编码器及聚合器，L2 归一化描述子，cosine 排名。
- 只用 WGS84 大地线中心距离报告 25/50/100 m 下 Recall@1/5/10 和 Top1 距离。
  此指标不等同于原评估入口的半图像半径阈值。
- 不需要 h_cover_m 或 aff2d_corrected；原始 GPS 精度和固定裁剪尺度会影响结果。

产物写入 `gen_fm_exps/xmu_xy/<UTC时间戳>/`：config.yaml、model_options.yaml、
gallery.pt、retrieval.npz、metrics.json、completed.json。
metadata 模式额外保存 query_features.pt；search 模式逐批计算 36 方向响应，
不保存全部方向的查询特征。retrieval.npz 额外保存 Top10 估计方向、真实方向、
Top1 方向误差及旋转网格（单位度，方向沿用 rotdeg_fm_north_anticlock 约定）。
`retrieval.npz` 包含 query 名称、经纬度、图库中心、Top10 索引/相似度/距离。
运行日志放 `gen_fm_exps/logs/xmu_xy/`。

快速验证可加 `--max-queries 8 --stride-m 200`，其结果不能当作全量评估。

`preflight.py` 是早期针对训练加载器的只读检查，报告中的缺字段仅阻止直接使用
训练加载器，不阻止本目录的独立 xy 评价。原始数据和公共实现未修改。

## 旋转响应聚合与排序

- `rot_weighting_argmax`（默认）：对每个 xy 的全部旋转，以
  `sum(exp(-sqrt(max(0, 2-2*cosine))/heading_temperature))` 聚合响应，
  实现使用等价且稳定的 logsumexp 分数。按聚合分数选不同 xy 的 Top10，
  然后在每个入选 xy 上取 cosine 最大的离散方向；方向不做加权平均。
- `rot_sum_argmax`：直接沿 rot 维度对原始 cosine 求和（含负值），
  按总和选不同 xy 的 Top10，再在每个位置取最大 cosine 的离散方向。
  不使用指数、softmax 或温度；top10_scores 保存 cosine 总和。
- `rot_xy_argmax`：将 rot×xy 展平，按 cosine 取 Top10 联合候选。
  同一 xy 的不同方向可重复占据名额，Recall@K 按这 K 个联合候选计算。
- `rot_argmax`：保留旧模式，每个 xy 先取最大 cosine，再排序选不同 xy。

`heading_temperature` 仅影响第一种模式的位置聚合；不影响离散方向 argmax。
metadata 模式忽略上述旋转搜索排序参数。retrieval.npz 的 top10_scores 保存实际
排序分数：指数聚合为 logsumexp，直接求和为 sum_cosine_over_rot，其他为 cosine；配置快照记录 score_type。

```bash
# 直接使用联合候选排序
/root/miniconda3/envs/neuloc_v0/bin/python experiments/xmu_stage1/evaluate_xy.py \
  --rot-retrieval-mode rot_xy_argmax
```

## 批量对照

编辑 `batch_config.yaml`：默认启用指数聚合、cosine 求和、rot×xy 联合排序、
xy 内最大值四组 36 朝向实验；各组通过 enabled 开关及 overrides 覆盖 config.yaml。
权重、数据和未覆盖参数统一继承基础配置，每组独立运行和建库。

```bash
# 生成配置，检查分组，不运行推理
/root/miniconda3/envs/neuloc_v0/bin/python experiments/xmu_stage1/run_batch.py --dry-run
# 运行全部启用的实验
/root/miniconda3/envs/neuloc_v0/bin/python -u experiments/xmu_stage1/run_batch.py
# 快速贯通四组（不是正式指标）
/root/miniconda3/envs/neuloc_v0/bin/python experiments/xmu_stage1/run_batch.py --max-queries 2 --stride-m 500
```

可用 --base-yaml 和 --batch-config 切换配置文件。未知覆盖项、重复名称报错。
按列表顺序运行，失败停止；每组配置及日志写到
`gen_fm_exps/logs/xmu_xy_batch/<时间戳>/`，manifest.json 记录状态、输出路径和指标，
summary.csv 随每组完成增量更新。评价产物写到各组 output_root 下的
`batch/<批次时间戳>/<组名>/<运行时间戳>/`，已有结果不会覆盖。

batch_size 是上限：建库前向进一步受 max_forward_variants 限制；查询响应按最多
64 张切块，避免大 batch 的 36 方向响应矩阵和指数聚合中间量占用过多显存。
所有查询均会处理，切块不改变每张查询的候选集合和排序规则。

`rotation_mode: none` 表示按原始朝向直接编码中心方形裁剪：既不使用记录朝向转正，
也不做方向搜索，不报告估计方向误差。用于 2DoF 模型的原始朝向评价。
