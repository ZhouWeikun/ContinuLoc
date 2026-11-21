"""Retained Stage 2 training experiment; reuse the shared command adapter."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from continuloc.cli.train_neural_field import main

if __name__ == "__main__":
    main(default_yaml="configs/neural_field_continuscenes.yaml")
