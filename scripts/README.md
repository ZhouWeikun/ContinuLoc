# 通用启动入口

从仓库根目录，在现有训练环境中执行：

```bash
python experiments/train_stage1/run.py --p_yaml configs/stage1_visual_encoder_visloc.yaml --selected_scene_names 03 04 08 11
python scripts/train_stage2.py --p_yaml configs/stage2_INGP_visloc.yaml --selected_scene_name 03
python scripts/localize_stage3.py --stage3_yaml configs/stage3_ingp_minimal.yaml --stage2_opts_yaml /path/to/stage2/opts.yaml
python scripts/update_data_splits.py --help
```

训练和推理需要相应数据、依赖及检查点；这里仅负责参数解析与调用。
实验专用启动和数据对齐可视化见 `experiments/`。旧 `trainers/` 中的三个入口
仅用于兼容既有命令。Stage 1 使用 `experiments/train_stage1/run.py`；其他通用命令使用本目录入口。
