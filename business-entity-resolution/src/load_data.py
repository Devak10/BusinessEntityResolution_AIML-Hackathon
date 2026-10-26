"""
Data Loading Module for Business Entity Resolution (Step 3)
===========================================================

Provides reusable, validated data loading functions for TRAIN and TEST partitions.
Strictly requires tab-separated TSV files and validates column schemas upon loading.

Schemas:
- Sources (1, 2, 3): entity_id, business_name, business_address, country
- Ground Truth:     source1_entity_id, matched_entity_ids
"""

from pathlib import Path
import pandas as pd

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def find_dataset_dir(custom_path=None) -> Path:
    """
    Locates the dataset directory automatically across standard workspace locations.
    """
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists():
            return p

    # 1. Check relative to current file in business-entity-resolution/src
    candidate1 = Path(__file__).resolve().parent.parent / "dataset"
    if candidate1.exists():
        return candidate1

    # 2. Check workspace root dataset
    candidate2 = Path(__file__).resolve().parent.parent.parent / "dataset"
    if candidate2.exists():
        return candidate2

    raise FileNotFoundError("Could not locate the dataset directory.")


def verify_columns(df: pd.DataFrame, expected: list[str], dataset_name: str):
    """
    Verifies that the DataFrame contains exactly the expected columns.
    Raises ValueError with details if schema does not match.
    """
    actual = list(df.columns)
    if actual != expected:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        raise ValueError(
            f"Schema mismatch in {dataset_name}!\n"
            f"  Expected columns: {expected}\n"
            f"  Actual columns:   {actual}\n"
            f"  Missing columns:  {list(missing)}\n"
            f"  Extra columns:    {list(extra)}"
        )


def load_train_data(dataset_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all four training datasets from dataset/train/*.tsv using tab delimiter.

    Returns:
        tuple: (s1_train, s2_train, s3_train, gt_train)
    """
    base_dir = find_dataset_dir(dataset_dir)
    train_dir = base_dir / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    s1_path = train_dir / "source1_train.tsv"
    s2_path = train_dir / "source2_train.tsv"
    s3_path = train_dir / "source3_train.tsv"
    gt_path = train_dir / "ground_truth_train.tsv"

    for p in [s1_path, s2_path, s3_path, gt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required training file: {p}")

    # Load with pandas strictly using sep="\t"
    s1_train = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    s2_train = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    s3_train = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    gt_train = pd.read_csv(gt_path, sep="\t", keep_default_na=False)

    # Validate column schemas
    verify_columns(s1_train, EXPECTED_SOURCE_COLUMNS, "source1_train.tsv")
    verify_columns(s2_train, EXPECTED_SOURCE_COLUMNS, "source2_train.tsv")
    verify_columns(s3_train, EXPECTED_SOURCE_COLUMNS, "source3_train.tsv")
    verify_columns(gt_train, EXPECTED_GT_COLUMNS, "ground_truth_train.tsv")

    return s1_train, s2_train, s3_train, gt_train


def load_test_data(dataset_dir=None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads all four testing datasets from dataset/test/*.tsv using tab delimiter.

    Returns:
        tuple: (s1_test, s2_test, s3_test, gt_test)
    """
    base_dir = find_dataset_dir(dataset_dir)
    test_dir = base_dir / "test"

    if not test_dir.exists():
        raise FileNotFoundError(f"Testing directory not found at: {test_dir}")

    s1_path = test_dir / "source1_test.tsv"
    s2_path = test_dir / "source2_test.tsv"
    s3_path = test_dir / "source3_test.tsv"
    gt_path = test_dir / "ground_truth_test.tsv"

    for p in [s1_path, s2_path, s3_path, gt_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required testing file: {p}")

    # Load with pandas strictly using sep="\t"
    s1_test = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    s2_test = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    s3_test = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    gt_test = pd.read_csv(gt_path, sep="\t", keep_default_na=False)

    # Validate column schemas
    verify_columns(s1_test, EXPECTED_SOURCE_COLUMNS, "source1_test.tsv")
    verify_columns(s2_test, EXPECTED_SOURCE_COLUMNS, "source2_test.tsv")
    verify_columns(s3_test, EXPECTED_SOURCE_COLUMNS, "source3_test.tsv")
    verify_columns(gt_test, EXPECTED_GT_COLUMNS, "ground_truth_test.tsv")

    return s1_test, s2_test, s3_test, gt_test
