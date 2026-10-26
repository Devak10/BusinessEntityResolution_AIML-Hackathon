"""
Root runner for verify_dataset.py.
Delegates to business-entity-resolution/src/verify_dataset.py.
"""
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from verify_dataset import run_verification

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    p1 = current_dir.parent / "business-entity-resolution" / "dataset"
    if p1.exists():
        run_verification(p1)
    p2 = current_dir.parent / "dataset"
    if p2.exists() and p2 != p1:
        run_verification(p2)
