# Refine after retrieval

从本地 `master` 恢复并适配的保留实验，来源见 [SOURCE.md](SOURCE.md)。
流程：已保存 Top-K → GIM 匹配 → RANSAC 相似变换 → 按内点数重排 → 修正 Top-1 的位置、方向和尺度。
默认匹配前 10 个候选，至少 8 个内点才修正；失败时保留重排后的坐标。

## YAML 批量运行（推荐）

修改同目录 `refine_batch_config.yaml`，然后执行：

```bash
conda run --no-capture-output -n gim python experiments/refine_after_retrieval/run_batch.py
```

当前预设指向已完成的 `rotWeightingN36_ep24` 检索结果，处理全部 9 个场景的 test 集，
匹配 Top-5、按内点数重排并修正 Top-1。无需再手工拼接 bundle 参数。

- `retrieval_dir`：评估结果目录或其 `retrieval/` 子目录。
- `split`：`test`、`val` 或 `all`，后者分别处理两个划分。
- `scenes`：空列表为全部场景，或指定 `[visloc_01, visloc_02]`；支持唯一简写 `["01", "02"]`（须加引号），重名时必须使用完整名称。
- `matcher_model`、`topn_match`、`min_inliers_for_refine`：匹配与修正参数。
- `limit_queries`：`null` 为全部，`2` 为每个场景/划分先处理两张。
- `save_intermediates`：保存图块和匹配明细。

每组以 `name`、`enabled`、`overrides` 定义，独立继承 `config.yaml`；可以添加多组比较。
相对数据路径以仓库根目录为基准。入口使用当前 Python 环境，因此应在 `gim` 环境运行。
`--dry-run` 只校验输入并预览命令，不加载模型，也不启动匹配。
支持 `--base-yaml` 和 `--batch-config` 指定其他配置文件。

结果：`gen_fm_exps/refine_after_retrieval/<name>/<时间>/gim_refine_summary.csv`。
其中 `refine_dist_median_meter` 是每个场景/划分全部 query 的最终 Top-1 定位中位数误差（米），
包含匹配失败后回退的 query；另有 `baseline_dist_median_meter`、`rerank_dist_median_meter` 供比较。
运行日志与解析后的 YAML、输入列表保存在 `logs/refine_after_retrieval_batch/<时间>/<name>/`。
批量任务顺序执行，失败即停止。

## 单次命令行运行

从项目根目录运行；使用已有 `gim` conda 环境和相邻的 `../pyproj_gim/` 实现。
权重默认使用该实现的 `/home/data/zwk/pyproj_gim/weights`；可用 `--weights-dir` 指定。

```bash
conda run -n gim python experiments/refine_after_retrieval/run.py \
  --bundle-path /absolute/path/to/retrieval_bundle.pt \
  --matcher-model gim_dkm --topn-match 5 --limit-queries 2 --save-intermediates
```

确认少量样本后，移除 `--limit-queries` 运行全部 query。
支持 `gim_roma`，以及旧版 `--summary-csv` 批处理输入；`--bundle-path` 可重复。
只读取自己生成或可信的 PyTorch bundle。

每次运行创建 `gen_fm_exps/refine_after_retrieval/<UTC时间>/`，包含运行参数、汇总 CSV，
以及每个输入的 `gim_refine_report.json`、`gim_refine_details.pt`。
`--save-intermediates` 额外保存 query、卫星图块和原始匹配数据。
重排和 refine 只作用于指定候选前缀，最终仅修正 Top-1，其他候选保留。

## 从当前 eval_stage1 接入

在 `experiments/eval_stage1` 的评估 YAML 中设置顶层 `export_retrieval: true`，
然后用该实验原有入口运行。默认关闭，不改变原有评估指标。
输出新增 `retrieval/<scene>__val/retrieval_bundle.pt` 和
`retrieval/<scene>__test/retrieval_bundle.pt`（仅生成非空划分）。将其中一个路径传给上面的入口。

Bundle 包含 `[N,K,4]` 的 `coords_topk`、`[N,4]` 的 `coords_gt`、阈值与 K 列表，
以及 query 原始路径、配置快照路径和 checkpoint 路径。坐标顺序为归一化 row、col、弧度方向、尺度。
旋转搜索保存每个候选的估计方向；使用已知方向对齐时保存 query 的已知方向。
因此开启 `val_query_rot2uniform` 的检索属于使用已知方向的设置，应与完全未知方向的实验区分。
默认沿用当前评估的位置阈值；方向、尺度联合阈值通过下面的重新统计指定。

旧版 Stage1 bundle 仍支持；无需恢复旧训练器。仅有 metrics.json 的旧评估需要重新运行以导出候选。

## 重算指标

编辑 `recall_config.yaml` 中的示例阈值后运行（无需再次匹配）：

```bash
conda run -n gim python experiments/refine_after_retrieval/export_matching_refine_recall_from_details.py \
  --details-root gen_fm_exps/refine_after_retrieval/<UTC时间> \
  --output-csv gen_fm_exps/refine_after_retrieval/<UTC时间>/recall.csv \
  --long-output-csv gen_fm_exps/refine_after_retrieval/<UTC时间>/recall_long.csv
```

## 验证

```bash
conda run -n gim python tmep/refine_after_retrieval/check_refine.py
conda run -n gim python tmep/refine_after_retrieval/check_stage1_export.py
```

构建期检查保留在 `tmep/refine_after_retrieval/`。覆盖重排、平移、方向/尺度修正、失败回退、
旧子集规则、显式 query 对齐、指标计算和输出防覆盖。未执行全数据集 GIM 实验。

已验证 CLI、运行依赖导入及上述 CPU 检查。当前 gim 环境报告 torchvision/xFormers 的可选 CUDA 扩展版本警告；尚未验证真实 GIM GPU 匹配。
