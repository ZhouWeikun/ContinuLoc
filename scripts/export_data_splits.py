"""Export fixed JSON train/test assignments without modifying source CSVs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from continuloc.engine.compat_runtime import preload_conda_libstdcpp
preload_conda_libstdcpp()
from continuloc.datasets.fixed_splits import export_fixed_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    index = export_fixed_splits(args.manifest, args.output_dir)
    for scene in index['scenes']:
        print(scene['scene'], {name: profile['eligible_count'] for name, profile in scene['profiles'].items()})


if __name__ == '__main__':
    main()
