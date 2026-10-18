# Stage 3 推理实验

固定流程：粗搜索 → 种群收缩 → CMA 局部精化。复用 `continuloc` 的 HDS 实现，不启用历史阶段跳过或梯度优化分支。

## 批量运行

在 `stage3_batch_config.yaml` 中填写每组的 `load_stage2_ckpt`、对应的 `inherit_stage2_yaml`，然后设置 `enabled: true`。
每组对应一个 Stage 2 单场景模型。`selected_scene_name` 可留空自动继承；显式指定时必须与 Stage 2 场景一致（VisLoc 编号也支持）。Stage 1 权重默认从 Stage 2 opts 追溯。

在项目根目录、已有 Stage 2 运行环境中执行：

```bash
python experiments/inference_stage3/run_batch.py --dry-run
python experiments/inference_stage3/run_batch.py
```

支持 `--batch-config`（别名 `--cfg-var-file`）、`--base-yaml` 和 `--python`。每组支持 `name / enabled / base_yaml / overrides`，独立继承基础配置；参数支持唯一短名或 `inference_setting.hds_stage3_cma_iters` 这样的完整路径。未知参数、重复名称或重复覆盖报错。顺序执行，失败停止。

`--dry-run` 校验权重路径、Stage 2 场景和评估阈值，保存配置与 manifest，但不加载模型、不执行推理。所有启用组预检通过后才开始推理。未填写路径的模板不能 dry-run；默认关闭的组会跳过。

## 单组运行

复制 `config.yaml` 并填写权重路径，然后执行：

```bash
python experiments/inference_stage3/run.py --config /path/to/config.yaml --dry-run
python experiments/inference_stage3/run.py --config /path/to/config.yaml
```

相对数据/权重路径和 batch 中的 `base_yaml` 相对项目根目录。CLI 指定的配置文件路径相对当前目录。
网络、数据与场景继承 Stage 2 opts，Stage 3 显式配置优先；硬件使用本实验配置。不要手写 `scenes_setting` 覆盖其他场景。`hds_eval_n_samples: null` 表示全部评估 query；冒烟验证可设为 1。限制数量取评估数据前 N 个，可能只有 val，不能据此报告完整 test 性能。

## 参数与指标

- `hds_stage1_*`：粗搜索网格、旋转/尺度采样、候选比例与上限。
- `hds_stage2_*`：种群收缩轮数、粒子数量、搜索半径、elite 与存活比例。
- `hds_stage3_*`：CMA 模式数量、种群、迭代与提前停止参数。
- `hds_score_chunk_size`：评分分块大小；不改变搜索阶段。

这里的 stage1/2/3 是推理内部阶段，区别于模型训练的 Stage 1/2。
评估阈值来自 `configs/stage3_recall_thresholds.yaml`：`stage3_recall_cfg: per_scene` 按场景选择，未列出的场景回退 `default: cfg01`（相对距离系数 0.55、旋转 5.5°、尺度比 1.2）。也可显式指定 cfg01 等命名配置。预检打印原始阈值，实际报告记录换算后的阈值。搜索半径与评估阈值分别管理。
保留 val/test 分开统计，各阶段提供距离、距离+旋转、距离+旋转+尺度 Recall@K 和误差统计；误差字段沿用核心报告的坐标单位，不能一律按米解读。

## 输出

- `gen_fm_exps/inference_stage3/<run_id>/`：`stage3_progressive_recall_report.json`、`predictions.pt`（各阶段坐标/分数、GT、评估 split 标签）、`timing.json`。
- `gen_fm_exps/logs/inference_stage3/<run_id>/`：提交给推理器的 `config.yaml`、`manifest.json`、`console.log`，以及核心运行日志。最终生效配置由核心在日志中打印。
- `gen_fm_exps/logs/stage3_batch/<timestamp>/summary.json`：逐组状态、指标、路径及进程总耗时；模型搜索耗时见各组 `timing.json`。

自动输出路径包含时间戳。显式设置 `stage3_analysis_export_root` 时要求目录尚不存在，避免覆盖历史结果。
构建期 CPU 校验：`python tmep/inference_stage3/check_launcher.py`。

结果目录同时保存 `config.yaml`（提交给推理器的 Stage 3 配置）、`stage2_opts.yaml`、可用时的 `stage1_opts.yaml`、`recall_thresholds.yaml`，以及与日志目录同步更新的 `manifest.json`。manifest 记录实际使用的 Stage 1/2 checkpoint 绝对路径、继承 YAML 原路径、配置快照路径、场景和运行状态。模型权重本身不复制；配置快照在启动推理前保存。`config.yaml` 与继承的 opts 快照共同描述测试配置，Stage 3 显式设置优先。
