"""
Dataset Inspection Script for Business Entity Resolution (Step 3)
=================================================================

Inspects TRAIN and TEST datasets loaded via load_data.py.
Displays:
- Dataset shapes
- Column information (types, missing counts, unique counts)
- Dynamic country distributions via value_counts()
- Sample records from all sources
- Data quality verification (missing fields, duplicates, prefixes, singletons)
- Train/Test consistency & overlap verification
"""

import sys
from pathlib import Path
import pandas as pd

# Support running directly or as module
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from load_data import load_train_data, load_test_data


def display_column_info(df: pd.DataFrame, title: str):
    """
    Displays column names, data types, missing counts, and unique counts.
    """
    print(f"\n{title}")
    print(f"{'-' * len(title)}")
    print(f"{'Column':<20} {'Type':<12} {'Missing':<10} {'Unique':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*10} {'-'*10}")

    for col in df.columns:
        col_type = str(df[col].dtype)
        # Check both NaN and blank string for text columns
        missing_count = int(df[col].isna().sum())
        if df[col].dtype == object and col != "matched_entity_ids":
            missing_count += int((df[col].astype(str).str.strip() == "").sum())

        unique_count = int(df[col].nunique())
        print(f"{col:<20} {col_type:<12} {missing_count:<10} {unique_count:<10}")


def check_data_quality(df: pd.DataFrame, name: str, expected_prefix: str = None) -> dict:
    """
    Performs data quality checks on a source dataset.
    """
    results = {}
    # 1. Missing entity IDs
    results["missing_entity_ids"] = int(df["entity_id"].isna().sum() + (df["entity_id"].astype(str).str.strip() == "").sum())
    # 2. Duplicate entity IDs
    results["duplicate_entity_ids"] = int(df["entity_id"].duplicated().sum())
    # 3. Missing business names
    results["missing_business_names"] = int(df["business_name"].isna().sum() + (df["business_name"].astype(str).str.strip() == "").sum())
    # 4. Missing business addresses
    results["missing_business_addresses"] = int(df["business_address"].isna().sum() + (df["business_address"].astype(str).str.strip() == "").sum())
    # 5. Missing country values
    results["missing_countries"] = int(df["country"].isna().sum() + (df["country"].astype(str).str.strip() == "").sum())
    # 6. Invalid entity ID prefixes
    if expected_prefix:
        invalid_prefix_count = int((~df["entity_id"].astype(str).str.startswith(expected_prefix)).sum())
        results["invalid_id_prefixes"] = invalid_prefix_count

    return results


def run_inspection():
    """
    Loads and inspects both train and test partitions.
    """
    s1_train, s2_train, s3_train, gt_train = load_train_data()
    s1_test, s2_test, s3_test, gt_test = load_test_data()

    print("========================================")
    print("DATASET INSPECTION")
    print("==================")

    # 1. Shapes
    print("\nTRAIN DATA")
    print(f"Source 1 shape     : {s1_train.shape}")
    print(f"Source 2 shape     : {s2_train.shape}")
    print(f"Source 3 shape     : {s3_train.shape}")
    print(f"Ground truth shape : {gt_train.shape}")

    print("\nTEST DATA")
    print(f"Source 1 shape     : {s1_test.shape}")
    print(f"Source 2 shape     : {s2_test.shape}")
    print(f"Source 3 shape     : {s3_test.shape}")
    print(f"Ground truth shape : {gt_test.shape}")

    # 2. Column Information
    print("\n========================================")
    print("COLUMN INFORMATION")
    print("==================")
    display_column_info(s1_train, "SOURCE 1 TRAIN")
    display_column_info(s2_train, "SOURCE 2 TRAIN")
    display_column_info(s3_train, "SOURCE 3 TRAIN")
    display_column_info(gt_train, "GROUND TRUTH TRAIN (Note: Empty matched_entity_ids = singletons)")

    display_column_info(s1_test, "SOURCE 1 TEST")
    display_column_info(s2_test, "SOURCE 2 TEST")
    display_column_info(s3_test, "SOURCE 3 TEST")
    display_column_info(gt_test, "GROUND TRUTH TEST (Note: Empty matched_entity_ids = singletons)")

    # 3. Country Distribution
    print("\n========================================")
    print("COUNTRY DISTRIBUTION")
    print("====================")
    print("\nSource 1 TRAIN:")
    print(s1_train["country"].value_counts().to_string())

    print("\nSource 2 TRAIN:")
    print(s2_train["country"].value_counts().to_string())

    print("\nSource 3 TRAIN:")
    print(s3_train["country"].value_counts().to_string())

    print("\nSource 1 TEST:")
    print(s1_test["country"].value_counts().to_string())

    print("\nSource 2 TEST:")
    print(s2_test["country"].value_counts().to_string())

    print("\nSource 3 TEST:")
    print(s3_test["country"].value_counts().to_string())

    # 4. Sample Records
    print("\n========================================")
    print("SAMPLE RECORDS (FIRST 5 ROWS)")
    print("=============================")

    print("\n--- Source 1 TRAIN (First 5) ---")
    print(s1_train.head(5).to_string(index=False))

    print("\n--- Source 2 TRAIN (First 5) ---")
    print(s2_train.head(5).to_string(index=False))

    print("\n--- Source 3 TRAIN (First 5) ---")
    print(s3_train.head(5).to_string(index=False))

    print("\n--- Ground Truth TRAIN (First 5) ---")
    print(gt_train.head(5).to_string(index=False))

    print("\n--- Source 1 TEST (First 5) ---")
    print(s1_test.head(5).to_string(index=False))

    print("\n--- Source 2 TEST (First 5) ---")
    print(s2_test.head(5).to_string(index=False))

    print("\n--- Source 3 TEST (First 5) ---")
    print(s3_test.head(5).to_string(index=False))

    print("\n--- Ground Truth TEST (First 5) ---")
    print(gt_test.head(5).to_string(index=False))

    # 5. Data Quality Checks
    print("\n========================================")
    print("DATA QUALITY CHECKS")
    print("===================")

    sources = [
        ("Source 1 Train", s1_train, "S1-"),
        ("Source 2 Train", s2_train, "S2-"),
        ("Source 3 Train", s3_train, "S3-"),
        ("Source 1 Test", s1_test, "S1-"),
        ("Source 2 Test", s2_test, "S2-"),
        ("Source 3 Test", s3_test, "S3-"),
    ]

    for name, df, prefix in sources:
        res = check_data_quality(df, name, expected_prefix=prefix)
        print(f"\n{name}:")
        print(f"  Missing entity IDs       : {res['missing_entity_ids']}")
        print(f"  Duplicate entity IDs     : {res['duplicate_entity_ids']}")
        print(f"  Missing business names   : {res['missing_business_names']}")
        print(f"  Missing addresses        : {res['missing_business_addresses']}")
        print(f"  Missing country values   : {res['missing_countries']}")
        print(f"  Invalid ID prefixes      : {res['invalid_id_prefixes']}")
        assert res["missing_entity_ids"] == 0, f"Quality failure in {name}: Missing entity IDs!"
        assert res["duplicate_entity_ids"] == 0, f"Quality failure in {name}: Duplicate entity IDs!"
        assert res["missing_business_names"] == 0, f"Quality failure in {name}: Missing business names!"
        assert res["missing_business_addresses"] == 0, f"Quality failure in {name}: Missing addresses!"
        assert res["missing_countries"] == 0, f"Quality failure in {name}: Missing countries!"
        assert res["invalid_id_prefixes"] == 0, f"Quality failure in {name}: Invalid ID prefixes!"

    # Ground truth specific quality checks
    gt_train_empty = int((gt_train["matched_entity_ids"].astype(str).str.strip() == "").sum())
    gt_test_empty = int((gt_test["matched_entity_ids"].astype(str).str.strip() == "").sum())

    print("\nGround Truth Quality:")
    print(f"  Train: Missing source1_entity_ids : {gt_train['source1_entity_id'].isna().sum()}")
    print(f"  Train: Duplicate source1_ids      : {gt_train['source1_entity_id'].duplicated().sum()}")
    print(f"  Train: Empty matched IDs (singletons): {gt_train_empty}")
    print(f"  Test : Missing source1_entity_ids : {gt_test['source1_entity_id'].isna().sum()}")
    print(f"  Test : Duplicate source1_ids      : {gt_test['source1_entity_id'].duplicated().sum()}")
    print(f"  Test : Empty matched IDs (singletons): {gt_test_empty}")

    assert gt_train["source1_entity_id"].isna().sum() == 0
    assert gt_test["source1_entity_id"].isna().sum() == 0
    assert not gt_train["source1_entity_id"].duplicated().any()
    assert not gt_test["source1_entity_id"].duplicated().any()

    # 6. Train / Test Consistency Checks
    print("\n========================================")
    print("TRAIN / TEST CONSISTENCY")
    print("========================")

    # Disjoint ID sets
    assert set(s1_train["entity_id"]).isdisjoint(set(s1_test["entity_id"])), "Overlap in Source 1 IDs!"
    assert set(s2_train["entity_id"]).isdisjoint(set(s2_test["entity_id"])), "Overlap in Source 2 IDs!"
    assert set(s3_train["entity_id"]).isdisjoint(set(s3_test["entity_id"])), "Overlap in Source 3 IDs!"
    print("Overlap check (Source 1) : 0 overlapping IDs (PASSED)")
    print("Overlap check (Source 2) : 0 overlapping IDs (PASSED)")
    print("Overlap check (Source 3) : 0 overlapping IDs (PASSED)")

    # Identical column schemas
    assert list(s1_train.columns) == list(s1_test.columns)
    assert list(s2_train.columns) == list(s2_test.columns)
    assert list(s3_train.columns) == list(s3_test.columns)
    assert list(gt_train.columns) == list(gt_test.columns)
    print("Schema alignment         : Identical columns between Train and Test (PASSED)")

    print("\nTRAIN/TEST CONSISTENCY: PASSED")
    print("========================================\n")


if __name__ == "__main__":
    run_inspection()
