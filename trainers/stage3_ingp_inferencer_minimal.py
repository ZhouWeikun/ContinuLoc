"""Compatibility entrypoint; prefer scripts/localize_stage3.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuloc.engine.stage3_ingp_inferencer_minimal import *

if __name__ == "__main__":
    from scripts.localize_stage3 import main
    main()
