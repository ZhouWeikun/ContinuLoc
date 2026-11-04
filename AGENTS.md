# 文件组织规则

遵循 `docs/project_layout.md`。

- 默认将新实验、诊断、功能构建期检查、可视化和一次性过程脚本放到 `tmep/<任务名>/`。目录名为 `tmep`。
- 本公开分支 `continuloc_v2_shared` 的 `experiments/` 仅保留 `train_stage1/`、`train_stage2/` 和 `inference_stage3/`；不得仅凭复用价值自行增加实验。
- 公共实现放 `continuloc/`，通用启动入口放 `scripts/`，共享配置放 `configs/`。公共包不得依赖临时或实验目录。
- 保留有用断言、运行说明和审计信息；临时目录不是自动删除授权。实验结果、诊断输出和可视化等产物统一放 `gen_fm_exps/`，不再使用 `results/` 或 `res_from_exps/` 作为新输出目录；检查点放 `gen_fm_exps/ckpts/`，日志放 `gen_fm_exps/logs/`。
