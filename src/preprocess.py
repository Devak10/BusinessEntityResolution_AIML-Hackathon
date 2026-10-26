"""
Root runner / entrypoint for preprocess.py.
Delegates to business-entity-resolution/src/preprocess.py.
"""
import sys
from pathlib import Path

project_src = Path(__file__).resolve().parent.parent / "business-entity-resolution" / "src"
sys.path.insert(0, str(project_src))

from preprocess import (
    normalize_text,
    remove_synthetic_markers,
    normalize_business_name,
    normalize_address,
    normalize_country,
    preprocess_source_dataframe,
)

__all__ = [
    "normalize_text",
    "remove_synthetic_markers",
    "normalize_business_name",
    "normalize_address",
    "normalize_country",
    "preprocess_source_dataframe",
]
