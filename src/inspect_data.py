"""
Root runner / entrypoint for inspect_data.py.
Delegates to business-entity-resolution/src/inspect_data.py.
"""
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from inspect_data import run_inspection

if __name__ == "__main__":
    run_inspection()
