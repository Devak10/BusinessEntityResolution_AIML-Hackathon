"""
Root runner for split_dataset.py.
Delegates to business-entity-resolution/src/split_dataset.py.
"""
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from split_dataset import main

if __name__ == "__main__":
    main()
