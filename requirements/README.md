# Environment setup

The reference environment uses Linux x86-64 (glibc 2.31 or later), Python 3.11, and an NVIDIA GPU with a driver that supports CUDA 12.1. Each stage includes the requirements of the previous stage. To add a later stage, install its requirements in the same environment.

| Stage | Requirements file | Adds |
|---|---|---|
| Train the visual encoder | [train_visual_encoder.txt](train_visual_encoder.txt) | PyTorch 2.1.0, torchvision 0.16.0, and data and training dependencies |
| Train the neural field | [train_neural_field.txt](train_neural_field.txt) | Kaolin 0.17.0 and dependencies for Kaolin Wisp |
| Evaluate pose inference | [eval_pose_inference.txt](eval_pose_inference.txt) | EvoTorch for CMA-ES pose refinement |

Neural field training and pose inference require the CUDA 12.1 Toolkit (`nvcc`), a compatible C++ compiler such as GCC 11, and the `libX11` runtime library used by Wisp's dependencies. Set `CUDA_HOME` to the toolkit directory if it is outside the default location.

To set up paper inference, run these commands from the repository root:

```bash
conda env create -f environment.yml
conda activate continuloc
python -m pip install -r requirements/eval_pose_inference.txt
python -m pip install --no-build-isolation -e .
```

For visual encoder training, substitute `requirements/train_visual_encoder.txt` and proceed to the [training instructions](../experiments/train_visual_encoder/README.md). For neural field training, substitute `requirements/train_neural_field.txt`. Both neural field training and pose inference also require [Kaolin Wisp](https://kaolin-wisp.readthedocs.io/en/latest/pages/install.html).

The Wisp patch makes the NeRF configuration importable, supports Python 3.11, and provides the YAML parser used to load the supplied hash grid configurations. The patch is distributed under the [NVIDIA Source Code License](kaolin-wisp-LICENSE.txt). From the ContinuLoc root, choose a path for the Wisp checkout and run:

```bash
export KAOLIN_WISP_ROOT=/path/to/kaolin-wisp
git clone https://github.com/NVIDIAGameWorks/kaolin-wisp.git "$KAOLIN_WISP_ROOT"
git -C "$KAOLIN_WISP_ROOT" checkout 931707e50f1511fdb4af55eeb4aed4df23b7c2b1
git -C "$KAOLIN_WISP_ROOT" apply "$PWD/requirements/kaolin-wisp.patch"
python -m pip install --no-build-isolation -e "$KAOLIN_WISP_ROOT"
```

Keep `KAOLIN_WISP_ROOT` set when running neural field training or inference.
