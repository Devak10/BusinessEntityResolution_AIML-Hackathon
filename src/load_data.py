"""
Root runner / entrypoint for load_data.py.
Delegates to business-entity-resolution/src/load_data.py.
"""
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from load_data import (
    load_train_data,
    load_test_data,
    find_dataset_dir,
    verify_columns,
    EXPECTED_SOURCE_COLUMNS,
    EXPECTED_GT_COLUMNS,
)

__all__ = [
    "load_train_data",
    "load_test_data",
    "find_dataset_dir",
    "verify_columns",
    "EXPECTED_SOURCE_COLUMNS",
    "EXPECTED_GT_COLUMNS",
]
