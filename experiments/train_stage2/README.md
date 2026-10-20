# Stage 2 坐标特征场训练

保留的 Stage 2 训练实验，复用 `continuloc/engine/stage2_INGP_minimal.py`。
已有精简训练路径与 master 同源：冻结 Stage 1 视觉编码器和聚合器并保持 eval 模式，
训练 HashGrid、坐标条件 MLP（以及配置启用的 hash_lod_aggregator），
用卫星裁剪和 UAV 图像的视觉描述子监督坐标特征，损失为 MSE × 1000。
保留当前分支的 val/test 分区指标实现，不引入 master 的 gallery 缓存和分析工具。

每次仅训练一个场景的特征场。Stage 1 可以使用多场景训练的模型，Stage 2 需逐场景运行，
分别命名和保存检查点。多个场景会报错，避免只训练首个场景。

从项目根目录，在安装了 PyTorch、Wisp/CUDA 扩展和数据依赖的训练环境中执行。
先填写配置中的 `load_stage1_ckpt` 和 `inherit_stage1_yaml`，也可通过命令行传入：

```bash
python experiments/train_stage2/run.py \
  --p_yaml experiments/train_stage2/config.yaml \
  --selected_scene_name 03 \
  --load_stage1_ckpt /path/to/stage1/epoch049.pth \
  --inherit_stage1_yaml /path/to/stage1/opts.yaml \
  --exp_name_override stage2_visloc03
```

不传 `--p_yaml` 默认使用上述 VisLoc 配置。Wingtra 使用 `config_wingtra.yaml`，
`--selected_scene_name` 可选 `zurich` 或 `zuchwil`。数据路径沿用共享配置，
需确保本机数据存在。`load_stage1_ckpt` 必须是权重文件，不能填写 `opts.yaml`；
`inherit_stage1_yaml` 应对应同一次 Stage 1 训练，继承网络结构及数据设置。
Stage 2 YAML 中显式指定的字段优先于继承字段。

断点续训在上述命令后追加 `--load2train /path/to/stage2/epoch009.pth`。
仅测试则追加 `--test_only --load2test /path/to/stage2/epoch049.pth`。
Stage 2 检查点仅保存可训练的场模块，因此续训、测试仍需要原 Stage 1 权重和配置。

日志写入 `gen_fm_exps/logs/<exp_name>/`，检查点写入
`gen_fm_exps/ckpts/<exp_name>/`。不同运行请设置不同实验名称。
通用入口 `scripts/train_stage2.py` 和旧 `trainers/stage2_INGP_minimal.py` 继续可用。
本目录配置将 Stage 1 路径留空，避免意外使用历史权重；启动前会检查必填路径。

## YAML 批量训练

编辑 `stage2_batch_config.yaml`，用与 `train_stage1/` 相同的格式管理多组实验：

- `name` 对应实验名称，必须唯一；`enabled: false` 跳过该组，默认 true。
- `base_yaml` 可为每组指定基础配置，相对项目根目录解析，优先于 `--base-yaml`。
- `overrides` 覆盖该组基础配置，未填写的参数独立继承；支持唯一简写键和点分嵌套键，
  例如 `grid_mlp_hidden_dim` 或 `network_setting.grid_mlp_hidden_dim`。
- 未知覆盖键、歧义键、重复名称和非布尔 `enabled` 会报错；实验名称只在 `name` 设置。

Stage 2的卫星尺度采样由 `data_setting` 控制：

- `sat_scale_sampling_mode`: `empirical`（UAV经验分布）、`uniform_linear`（raw scale均匀）或
  `uniform_log`（log scale均匀）。
- `sat_scale_jitter_enabled`: 是否在基础尺度上叠加随机扰动。
- `sat_scale_jitter_half_span_ratio`: 扰动半径占完整尺度跨度的比例；`0.05`表示
  `±5% * (scale_max - scale_min)`。关闭扰动时该数值不生效。

这些字段可在 `stage2_batch_config.yaml` 的 `overrides` 中逐组覆盖。文件末尾保留了
Zuchwil interval的 `uniform_log + no jitter` 对照配置，默认禁用，避免批量执行时误启动训练。

列表包含 VisLoc 03、Wingtra Zurich、Wingtra Zuchwil 三组独立训练示例。
先填写每组 `load_stage1_ckpt` 与对应的 `inherit_stage1_yaml`，不需要的组设为禁用。
每组仍只训练一个场景；可复制条目来比较不同场景、教师检查点或 MLP 参数。
Stage 1 数据划分默认从其 `opts.yaml` 继承。

预览并生成完整 YAML（不加载模型、不检查数据或权重是否存在、不启动训练）：

```bash
python experiments/train_stage2/run_batch.py --dry-run
```

顺序启动全部启用的实验：

```bash
python experiments/train_stage2/run_batch.py
```

支持 `--base-yaml`、`--batch-config`（别名 `--cfg-var-file`）和 `--output-dir`。
默认基础配置为本目录 `config.yaml`，批量列表为 `stage2_batch_config.yaml`。
完整配置默认保存到 `gen_fm_exps/logs/stage2_batch/<UTC时间戳>/`，不覆盖已有 YAML。
全部禁用时不创建输出目录。使用当前 Python 环境从项目根目录启动各组训练，
前一组成功退出后才执行下一组，任一组失败立即停止。
各组日志和检查点仍遵循其生成 YAML 中的 `dir2save_log`、`dir2save_ckpt`。

## Wisp / Kaolin 训练环境

若出现 `ModuleNotFoundError: No module named 'kaolin'`，说明当前 Python 未安装
Stage 2 HashGrid 所需的 Kaolin。Stage 1 能运行不代表该环境具备 Wisp/CUDA 扩展。
本机可显式使用已安装 Kaolin 的 `neuloc_wisp` 环境：

```bash
python experiments/train_stage2/run_batch.py \
  --python /root/miniconda3/envs/neuloc_wisp/bin/python
```

`--python` 仅指定训练子进程解释器，默认仍为当前 Python；也适用于 `--dry-run`。
单组训练可直接用该 Python 执行 `run.py`。不同环境的 PyTorch/CUDA 扩展需相互匹配，
不要通过添加另一环境的 site-packages 来混用 Kaolin。
Grid 初始化会优先加载当前环境的 `libstdc++`，避免系统旧运行库缺少
`GLIBCXX_3.4.29` 导致 Wisp 扩展导入失败。
