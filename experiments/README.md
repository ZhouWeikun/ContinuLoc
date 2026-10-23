# Experiments

仅保存用户明确指定长期保留的实验。未经用户指定的实验、诊断、验证和过程脚本默认放到 `tmep/<任务名>/`，不得自行提升到此目录。

- `train_stage2/`：Stage 2 单场景坐标特征场训练与测试，支持 VisLoc 和 Wingtra。
- `train_stage1/`：Stage 1 视觉编码器训练与测试入口，支持 VisLoc 和 Wingtra。
- `stage3_search_projection/`：保留的历史绘图源码；缺少旧分支依赖，暂不可运行。

生成结果放到 `gen_fm_exps/<run_id>/`。

- `refine_after_retrieval/`：检索结果经过 GIM 匹配、内点重排与 Top-1 位姿修正。

- `inference_stage3/`：固定粗搜索、种群收缩、CMA 精化的 Stage 3 单次与批量推理。
- `analyse_stage3/`：读取 Stage 3 已保存预测，导出方向箭头、位置/旋转/尺度误差、排序图库和逐样本统计。

- `xmu_stage1/`：冻结已有 Stage1 模型，在厦大翔安数据上建立特征库并评价二维位置检索。
