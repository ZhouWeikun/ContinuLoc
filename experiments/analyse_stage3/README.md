# analyse_stage3：Stage 3 离线错误分析

默认输入为 Zuchwil 20261019T034406528867Z 的已保存预测，scale=4、particles=8。
不重新推理、不修改权重、原始结果或现有实验配置。

```bash
/root/miniconda3/envs/neuloc_wisp/bin/python experiments/analyse_stage3/run.py \
  --config experiments/analyse_stage3/config.yaml
```

编辑 `config.yaml` 的 `run`、`output`、`satellite_index` 即可切换结果和底图；
也支持命令行 `--run`、`--output` 覆盖。配置内相对路径相对于项目根目录。
默认输出为 `gen_fm_exps/analyse_stage3/zuchwil_interval_scale4_m8/`。
相同输出目录重跑会更新同名分析产物；旧临时分析产物保持原位。
运行时保存 `analysis_config.yaml` 供复现。

场景数据路径和推理参数来自结果目录的 `config.yaml`，
划分名称来自 `stage2_opts.yaml`，阈值来自原始 recall 报告。
当前适配 Wingtra Zurich / Zuchwil 的投影坐标 CSV 和未旋转卫星栅格；
其他数据格式需另行适配。绘图文字、字体尺寸和图幅配置集中在 `plot.py` 顶部。

用浏览器打开输出的 `index.html`，可按位置、旋转、尺度误差排序，
并筛选“位置与旋转通过，仅尺度失败”的样本。全部 346 张图位于 `images/`，
逐样本结果为 `per_query.csv`，聚合统计为 `summary.json`。
每张图上方的文字标题可选中复制，包含文件名、Query 编号、原 CSV 数据行索引
（从 0 开始，不含表头）、最终误差及覆盖长度，方便记录后续筛选对象。
标题还包含 GT/场景中位数、GT/场景均值、预测/场景中位数、预测/场景均值；
图中 GT 列及三个预测阶段也显示对应比值，CSV 保存全部数值。
场景均值和中位数复用 `_resolve_sample_selection` 的样本选择，
统计范围是定义卫星数据集尺度分布的场景选中样本（空间边界过滤及训练/测试拆分之前），
不是仅对当前测试集求平均，也不是 `scale_ref_m` 归一化参考值。
网页顶部列出样本数、米制及归一化尺度统计值；
`summary.json` 保存统计口径、选择设置和源 CSV 的 SHA256。
这些统计根据当前 CSV 和运行时保存的配置重建；历史 CSV 如已被改动，
不能将其视为运行当时的完整快照。

## 图像与审计口径

- 五列：UAV 中心正方形、GT 卫星裁剪、粗搜索 Top1、模式筛选 Top1、最终 Top1。
- 红色箭头为 GT，蓝色为预测，箭头长度比为 GT_scale / pred_scale。
  GT 在预测裁剪范围外时，箭头也可能在图外；数值误差仍完整显示。
- 卫星底图由 `satellite_index` 指定，默认本场景的 SWISSIMAGE2018；图像为离线诊断重采样，不宣称逐像素复现网络预处理。
- 地理坐标与 CSV 一对一匹配，并断言位置、旋转、尺度参考值一致。
  尺度参考值由保存的 GT 与 CSV 覆盖长度反推并逐帧验证，结果为 120 m。
- 位置误差用原报告的 nrc2meter；尺度误差为 max(pred/GT, GT/pred)。
  原始有方向比值 pred/GT 单独输出，避免混淆高估和低估。
- 各阶段按保存分数选取 Top1，断言最终三个召回计数与原报告一致。
- `master_pose_arrows.py` 原样提取自 master
  `cfb8d0724bf68bcab2c888aff790bc9c126711a9` 的
  `scripts/visualization/draw_gt_arrow_on_stage3_preds.py`，
  复用其坐标投影、方向及箭头绘制函数。
  master 另有 `scripts/visualization/export_stage3_error_rank_visuals.py`，
  但依赖旧版 `stage3_triplets_summary.csv` 与 samples 目录，不能直接读取本次 predictions.pt。

## 默认 Zuchwil 结果的初步结论

位置通过 207/346，位置与旋转通过 173/346，三项通过 54/346。
互斥失败分类：位置失败 139，位置通过但旋转失败 34，
位置旋转通过但尺度失败 119，全部通过 54。

GT 覆盖长度范围 92.71–178.84 m，中位数 107.85 m；
粗搜索 Top1 覆盖长度已集中于 133.13–136.56 m；
最终 Top1 集中于 132.77–137.21 m，中位数 135.39 m。
因此尺度失败包含明显的尺度估计集中现象，并非仅由少数极端样本造成。

配置使用 `log_expectation`，在粗搜索中对尺度的对数作概率加权平均，
随后关闭粒子尺度采样及 CMA 尺度优化。若尺度权重缺乏区分度，
加权平均会集中到中间尺度；当前预测分布与此机制相符。
但保存文件不含完整尺度概率，尚不能据此确认权重变平的原因，
也不能直接认定为网络训练失败或实现错误。
要区分这些原因，下一步应导出各尺度得分，并以相同权重和样本对照
argmax 与 log_expectation、固定尺度与尺度优化的结果。

代表样本 136：位置误差 2.01 m，旋转误差 7.64°；
GT 覆盖 98.37 m，预测 134.49 m，尺度比 1.367，单独因尺度失败。
