# 文件组织

- `continuloc/`：可导入的公共实现。`datasets` 数据加载，`geometry` 几何变换，
  `sampling` 采样，`models`/`losses` 模型与目标，`evaluation` 指标，
  `engine` 训练、推理与运行管理，`localization` 定位搜索与精化，`config` 配置解析。
- `scripts/`：通用命令入口，不复制训练或数据处理实现。
- `configs/`：共享配置。数据清单仍保留在外部数据集目录中。
- `experiments/`：仅存放用户明确指定保留的实验，目前为 `stage3_search_projection/` 、`train_stage1/` 、`train_stage2/` 、`eval_stage1/` 、`refine_after_retrieval/` 、`inference_stage3/` 、`analyse_stage3/` 和 `xmu_stage1/`。
- `tmep/`：未经用户指定保留的实验、诊断、验证和过程脚本的默认位置；仅在用户明确指定后移入 experiments。旧 `temp/` 内容保留，新工作使用 `tmep/`。
- `gen_fm_exps/`：统一存放实验结果、评估与诊断输出、预测、数组、表格和可视化图片，按任务或 run_id 建立子目录。后续输出统一使用此命名，不再使用 `results/` 或 `res_from_exps/`。
- `gen_fm_exps/ckpts/`：存放检查点；`gen_fm_exps/logs/`：存放日志，与实验结果以相同 run_id 关联。
  历史运行产物保留原位；后续运行若沿用旧配置或脚本，应将结果输出路径显式改为 `gen_fm_exps/`。
- `data_splits/`：未来外部样本划分协议的版本化存放位置，当前尚未生成协议。

公共包不依赖 experiments、scripts、tmep 或旧 temp。已有自动断言保留在
`tmep/development_checks/`，不要求独立顶层 tests 目录。

## 兼容性

三个旧 `trainers/stage*_minimal.py` 入口转发到新命令，并保留训练器类导出。
`trainer_depends/configs` 是到 `configs` 的兼容链接，使旧启动命令和已保存
配置中的路径仍然有效。其他内部 Python 导入已迁移至 `continuloc`；外部脚本
若导入旧内部模块，需要按 `layout_migration.json` 更新。

本次仅调整源码结构，不改变数据筛选、划分或几何采样算法。
旧检查点若保存的是整个 Python 模型而非 state_dict，可能需要旧模块环境加载。
