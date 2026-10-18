# Stage1 evaluation

单次评估：

```bash
python experiments/eval_stage1/run.py --config experiments/eval_stage1/config.yaml
```

批量评估：

```bash
python experiments/eval_stage1/run_batch.py --dry-run
python experiments/eval_stage1/run_batch.py
```

编辑 `eval_batch_config.yaml`：每组有唯一的 name、enabled 和 overrides。
省略参数独立继承基础配置；删除覆盖项不代表禁用功能。
支持顶层 checkpoint/scenes/output_root 和 evaluation 参数简名或点分键
（例如 val_query_rot2uniform 或 evaluation.val_query_rot2uniform）。
未知键、重复名称/覆盖项报错。全部禁用不生成文件。
可指定 --base-yaml、--batch-config（别名 --cfg-var-file）及 --output-dir。
完整批量配置默认保存在 logs/eval_stage1_batch/<时间戳>/。
正常执行按顺序运行，失败即停止；dry-run 只生成配置和打印命令。

模型结构、场景及数据划分继承 checkpoint 同目录的 opts.yaml；
scenes: [] 使用全部场景。结果目录包含训练实验名、checkpoint、
评估名称和时间戳，保存配置快照、eval.log、增量 metrics.json 及成功标记
completed.json。日志包含标准输出、标准错误及异常，位于各自实验目录内。

## 独立的方向与尺度控制

不再使用 retrieval_mode，也不兼容旧模式或旧尺度字段。

- val_query_rot2uniform: true：使用记录朝向将 query 对齐北，只编码一个方向。
  query_rotation_step_deg 和 heading_temperature 完全忽略。
- val_query_rot2uniform: false：不使用记录朝向转正，按 query_rotation_step_deg
  生成整圈旋转候选；步长必须整除 360。由 rot_retrieval_mode 选择排序方式。
- val_query_scale2uniform：独立控制是否使用 query 尺度信息归一到场景均值。
  false 保留原尺度，在方向搜索中也不会强制归一。
- val_gallery_scale_mode: single：由 val_gallery_single_scale_level 决定尺度，
  当前仅支持 scene_mean；忽略两个 multi_scale 参数。
- val_gallery_scale_mode: multiscale：使用场景元数据中的尺度上下限，
  由 val_gallery_multi_scale_levels（至少 2）和
  val_gallery_multi_scale_spacing（linear 或 log）确定分箱中心。
  linear 为线性分箱中心，log 为对数空间分箱中心；不包含边界端点。
  忽略 single_scale_level。当前不提供手动尺度上下限覆盖项。
- val_overlap：gallery 裁剪重叠率，范围 [0, 1)。

所有方向/query 尺度/gallery 尺度组合均可运行。gallery 始终朝北。
多尺度中每个位置与尺度的组合都是独立候选；不跨尺度合并或按位置去重，
因此多个尺度候选可能占据 Top-K 中的多个名额。

方向搜索支持三种 rot_retrieval_mode（默认 rot_weighting）：

- rot_weighting：每个 xy 聚合全部方向的非线性响应，再选 TopN xy；
  在每个选中 xy 上加权估计方向。heading_temperature 仅在此模式生效，必须为正数。
- rot_argmax：每个 xy 取最大 cosine 及对应离散方向，再按最大值选 TopN xy。
- rot_xy_argmax：展平 xy × 方向，直接按 cosine 选 TopN 联合候选，不按 xy 去重。
  同一 xy 的不同方向可以重复占据 TopN 名额，Recall@K 也按这 K 个联合候选计算。

当前内部响应是 cosine，相似度最大等价于归一化特征距离最小。
后两种模式忽略 heading_temperature；统一朝北时上述三个搜索参数均忽略。
现有方向误差只评估排名第一候选。尺度规则不因本次排序模式改动。
YAML 按 xy、scale、rotation、performance 顺序组织。

rot_weighting 保持原评分公式：归一化特征距离为 sqrt(max(0, 2 - 2 cos))，
每个 gallery 候选对全部角度的 exp(-distance / temperature) 求和，
实现使用 float64 logsumexp 保持稳定排序。方向在获胜候选上计算：
以响应峰值角度为基准，用相同温度的 softmax 加权局部角度偏差，
再取负号转换为 rotdeg_fm_north_anticlock 约定。
分别报告 val/test 的 Recall、位置误差和方向误差；朝北模式不报告方向估计误差。
位置命中阈值保留现有场景半图像半径定义。

## 性能参数

- max_forward_variants: 1024：单次前向图像数上限，同时用于建库分块。
  36 个角度时每批 28 个 query / 1008 个候选；角度数超过上限时继续分块。
  未设置时仍使用 val_chunk_size 作为前向分块回退值。
- batchsize_uav: 32：CPU DataLoader 读取批量，与 GPU 前向批量独立，
  加载批次按顺序重新组合。num_worker_eval 控制读取进程数。
- 查询按批处理，gallery 特征、相似度及排名在 GPU；只回传指标所需信息。
  satmaps_on_cpu: true 将卫星大图留在 CPU。
- eval_cuda_memory_budget_gib: 22：独立入口将 PyTorch 分配器上限设为 21 GiB，
  留 1 GiB 给 CUDA 外部分配；超限报 OOM，并非整卡显存的绝对限制。
  多尺度增大 gallery 和响应矩阵，需要时降低 max_forward_variants。
- 1024 默认值基于 RTX 4090、当前 DINOv2/SALAD 和 224 输入测量。
  更大图像、模型或 gallery 需重新测量；保持 float32 网络推理。

默认基线为 ckpt10、query 朝北、不归一 query 尺度、gallery 使用场景均值。
2DoF/4DoF 模型使用同一套开关，训练 pair_alignment_mode 不选择评估策略。
