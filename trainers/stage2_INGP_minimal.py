"""Compatibility entrypoint; prefer scripts/train_stage2.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuloc.engine.stage2_INGP_minimal import *

if __name__ == "__main__":
    from scripts.train_stage2 import main
    main()
