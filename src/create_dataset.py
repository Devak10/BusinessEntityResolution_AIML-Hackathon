"""
Root runner / entrypoint for create_dataset.py.
Delegates to business-entity-resolution/src/create_dataset.py.
"""
import sys
from pathlib import Path

# Add business-entity-resolution/src to sys.path and execute
project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from create_dataset import main

if __name__ == "__main__":
    main()
