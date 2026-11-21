# Experiments

The experiment directories follow Sections 4.2–4.4 of the ContinuLoc paper.

| Component | Purpose | Guide |
|---|---|---|
| Learning Pose-Sensitive Visual Representations | Train the visual encoder, optionally across multiple scenes | [Visual encoder](train_visual_encoder/README.md) |
| Neural Compression of Pose-conditioned Semantic Fields | Train one semantic neural field per scene | [Neural field](train_neural_field/README.md) |
| Hierarchical Search over the Compatibility Landscape | Infer continuous UAV poses using the trained models | [Pose inference](eval_pose_inference/README.md) |

For pretrained-model evaluation, follow the [paper reproduction workflow](../README.md#reproduce-inference-with-the-supplied-models). The supplied bundle pairs checkpoints with their exact configuration and evaluation resources.

For new experiments, each directory provides `run.py`, `run_batch.py`, and one disabled batch YAML example. Set local paths and parameters, then enable the example. Copy an entry and give it a unique name to run additional experiments. All entries use the same ContinuScenes interface; choose scenes by name rather than separate dataset-specific templates.

Run commands from the repository root. Training logs and checkpoints are stored under `gen_fm_exps/logs/` and `gen_fm_exps/ckpts/`.
