import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from continuloc.datasets.update_split_columns_from_yaml import main

if __name__ == "__main__":
    main()
