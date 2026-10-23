# Stage 1 视觉编码器训练

用户指定长期保留的训练实验。`run.py` 负责加载配置、解析参数并调用
`continuloc/engine/stage1_visual_encoder_minimal.py` 中的公共训练器。

## VisLoc

在项目根目录、已安装训练依赖的环境中执行：

```bash
python experiments/train_stage1/run.py \
  --p_yaml configs/stage1_visual_encoder_visloc.yaml \
  --selected_scene_names 01 02 03 04 05 06 08 09 10 11 \
  --exp_name_override stage1_visloc_globalneg_grid100m
```

当前 manifest 未注册 07，因此命令显式排除它。

## Wingtra

```bash
python experiments/train_stage1/run.py \
  --p_yaml configs/stage1_visual_encoder_wingtra.yaml
```

不传 `--p_yaml` 时默认使用 VisLoc 配置。传 `--test_only` 仅执行测试，
测试检查点由配置的 `load2test` 指定。`--exp_name_override` 可以覆盖实验名称。

Wingtra 的 `data_setting.scale_filter_mode` 显式控制尺度过滤，两个场景使用同一选项：

| 值 | 过滤方式 | 归一化参考尺度 |
|---|---|---|
| `none` | 仅保留质量有效样本，不按尺度剔除 | 选中样本均值向上取到下一个 10 m |
| `percentile` | 质量有效样本中严格保留 P2～P99 | 固定 200 m，复现旧 Wingtra 分支 |
| `mad`（当前 Wingtra 默认） | modified-z ≤ 3.5，含保留区间端点；MAD 为零时保留近似中位数的样本 | 过滤前质量有效样本均值向上取到下一个 10 m |

`mad` 复现 master 的 `zurich` / `zuchwil` 场景实际进入的通用过滤分支，
不再依赖场景名是否含 `wingtra`。先进行尺度过滤与空间边界过滤，再生成划分。
显式 `split_source: csv` 若选中了被过滤的行会报错，不会静默修改 CSV。
模式随 opts.yaml 保存，并随 Stage 1 → Stage 2 → Stage 3 的 data 继承传递；
未配置新选项的旧 YAML 继续保持原行为，显式新选项优先于旧 `enable_scale_filter`。
复用此次未过滤训练的旧权重做评估时应使用其原配置（`none`），
切换 MAD 会改变样本与划分。恢复过滤口径不保证与 master 历史结果完全相同，
还需保持原始 CSV、划分、训练配置和权重一致。

共享配置继续保留在 `configs/`，避免后续阶段引用路径变化。
日志、检查点位置遵循 YAML 中的设置。旧 `trainers/stage1_visual_encoder_minimal.py`
入口继续转发到这里；原 `scripts/train_stage1.py` 已移动，不再作为启动入口。

## YAML 批量实验

编辑 `stage1_batch_config.yaml` 的 `experiments` 列表，统一管理 VisLoc 和 Wingtra：
每组 `name` 对应实验名称，`enabled: false` 跳过整组（默认 true），
`overrides` 覆盖基础 YAML，未填写参数独立继承基础配置。
每个数据集启用三组 softplus triplet：4DoF/alpha10、2DoF/alpha10、2DoF/alpha20；
InfoNCE/temperature0.05/4DoF 默认禁用。两类数据集依次独立训练，不是混合数据训练同一个模型。
Wingtra 使用 Zurich、Zuchwil 两个场景，采用与 VisLoc 对照一致的 generated/segment
划分（train 80%、val 5%、test 15%）和 50 epochs。CSV 的 aff2d_corrected
质量过滤仍生效，但历史 split 列及其空行不参与筛选。
全部禁用时直接退出，不生成配置或启动训练。

每组可设置 `base_yaml: configs/stage1_visual_encoder_wingtra.yaml`；
相对路径以项目根目录为基准，绝对路径也可使用。该字段优先于命令行 `--base-yaml`；
未指定时继续使用命令行基础配置（默认 VisLoc），覆盖键按该组基础配置校验。

支持 YAML 列表和点分嵌套键，例如
`network_setting.backbone_config.num_trainable_blocks: 2`。
简写键必须在基础 YAML 的一级配置组中唯一存在。场景编号请加引号。
未知覆盖键、重复实验名称及非布尔 enabled 值会报错。
实验名称只在 `name` 设置，不要在 overrides 中重复设置 exp_name。

先预览（生成每组完整 YAML，不启动训练）：

```bash
python experiments/train_stage1/run_batch.py --dry-run
```

确认后顺序运行：

```bash
python experiments/train_stage1/run_batch.py
```

可通过 `--base-yaml`、`--batch-config` 指定其他基础 YAML 和实验列表 YAML（`--cfg-var-file` 仍作为参数别名）。
默认将完整配置保存在 `gen_fm_exps/logs/stage1_batch/<时间戳>/`，可用 `--output-dir` 指定目录；
不覆盖已有 YAML。训练日志和检查点仍遵循各组 YAML 配置；当前列表显式设置为项目下的 `gen_fm_exps/logs/` 和 `gen_fm_exps/ckpts/`。
使用当前 Python 环境启动子进程，一组训练退出后再运行下一组，失败立即停止。
再次运行相同实验列表 会使用相同实验名称；请自行调整名称以区分实验。
