"""Compatibility entrypoint; prefer experiments/train_stage1/run.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuloc.engine.stage1_visual_encoder_minimal import *

if __name__ == "__main__":
    from experiments.train_stage1.run import main
    main()
