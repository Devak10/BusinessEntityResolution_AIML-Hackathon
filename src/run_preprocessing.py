"""
Root runner / entrypoint for run_preprocessing.py.
Delegates to business-entity-resolution/src/run_preprocessing.py.
"""
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from run_preprocessing import run_preprocessing_pipeline

if __name__ == "__main__":
    run_preprocessing_pipeline()
