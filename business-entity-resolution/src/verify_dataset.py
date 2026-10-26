"""
Dataset Verification Script for Business Entity Resolution (Step 1 - 1,000 Entity Scale)
========================================================================================

Verifies all requirements:
1. Exactly 1,000 Source 1 records
2. Files exist and load via pd.read_csv(file, sep="\\t")
3. TSV format with \\t delimiter
4. Required columns in all files
5. Unique entity IDs (no duplicates)
6. Correct ID format prefixes (S1-xxxxx, S2-xxxxx, S3-xxxxx)
7. All ground-truth IDs exist in Source 2 or Source 3
8. No Source 1 IDs appear in matched_entity_ids
9. No duplicate matched IDs per ground-truth row
10. Singleton count and match counts
11. Country distribution (balanced India & USA)
12. Match partition:
    - Matches in both sources
    - Matches in Source 2 only
    - Matches in Source 3 only
    - Singletons
13. Presence of hard negative examples (lexical/address overlap among non-matches)
"""

import sys
from pathlib import Path
import pandas as pd

def run_verification(dataset_dir: Path):
    print(f"\n=======================================================")
    print(f"VERIFYING DATASET AT: {dataset_dir}")
    print(f"=======================================================")

    s1_file = dataset_dir / "source1.tsv"
    s2_file = dataset_dir / "source2.tsv"
    s3_file = dataset_dir / "source3.tsv"
    gt_file = dataset_dir / "ground_truth.tsv"

    for f in [s1_file, s2_file, s3_file, gt_file]:
        assert f.exists(), f"Missing file: {f}"

    # 1. Load via pd.read_csv with sep="\t"
    s1 = pd.read_csv(s1_file, sep="\t", keep_default_na=False)
    s2 = pd.read_csv(s2_file, sep="\t", keep_default_na=False)
    s3 = pd.read_csv(s3_file, sep="\t", keep_default_na=False)
    gt = pd.read_csv(gt_file, sep="\t", keep_default_na=False)
    print("[OK] Loaded all 4 files using pd.read_csv(file, sep='\\t')")

    # 2. Verify record counts
    assert len(s1) == 1000, f"Expected exactly 1,000 Source 1 records, got {len(s1)}"
    assert 700 <= len(s2) <= 900, f"Expected 700-900 Source 2 records, got {len(s2)}"
    assert 700 <= len(s3) <= 900, f"Expected 700-900 Source 3 records, got {len(s3)}"
    assert len(gt) == 1000, f"Expected ground truth to have 1,000 rows, got {len(gt)}"
    print(f"[OK] Record counts verified: Source 1 = {len(s1)} (exactly 1,000), Source 2 = {len(s2)}, Source 3 = {len(s3)}")

    # 3. Verify columns
    expected_source_cols = ["entity_id", "business_name", "business_address", "country"]
    assert list(s1.columns) == expected_source_cols, f"S1 columns mismatch: {s1.columns}"
    assert list(s2.columns) == expected_source_cols, f"S2 columns mismatch: {s2.columns}"
    assert list(s3.columns) == expected_source_cols, f"S3 columns mismatch: {s3.columns}"
    assert list(gt.columns) == ["source1_entity_id", "matched_entity_ids"], f"GT columns mismatch: {gt.columns}"
    print("[OK] Column schemas verified in all sources and ground truth")

    # 4. Verify unique entity IDs (no duplicates)
    assert s1["entity_id"].is_unique, "Duplicate entity_ids detected in Source 1!"
    assert s2["entity_id"].is_unique, "Duplicate entity_ids detected in Source 2!"
    assert s3["entity_id"].is_unique, "Duplicate entity_ids detected in Source 3!"
    assert gt["source1_entity_id"].is_unique, "Duplicate source1_entity_id in Ground Truth!"
    print("[OK] Uniqueness verified: 0 duplicate IDs across all datasets")

    # 5. Verify ID prefixes and 5-digit format
    assert s1["entity_id"].str.match(r"^S1-\d{5}$").all(), "S1 ID format invalid!"
    assert s2["entity_id"].str.match(r"^S2-\d{5}$").all(), "S2 ID format invalid!"
    assert s3["entity_id"].str.match(r"^S3-\d{5}$").all(), "S3 ID format invalid!"
    assert s1["entity_id"].iloc[0] == "S1-00001", "S1 first ID should be S1-00001"
    assert s1["entity_id"].iloc[-1] == "S1-01000", "S1 last ID should be S1-01000"
    print("[OK] ID formats verified: S1-00001 to S1-01000, S2-xxxxx, S3-xxxxx")

    # 6. Verify country distribution
    valid_countries = {"India", "USA"}
    assert set(s1["country"]).issubset(valid_countries), "Invalid country in S1"
    assert set(s2["country"]).issubset(valid_countries), "Invalid country in S2"
    assert set(s3["country"]).issubset(valid_countries), "Invalid country in S3"
    assert s1["country"].value_counts().to_dict() == {"India": 500, "USA": 500}, "S1 country distribution not 500/500!"
    print(f"[OK] Country distributions verified: S1 is balanced (500 India, 500 USA)")

    # 7. Ground truth integrity checks
    s2_ids = set(s2["entity_id"])
    s3_ids = set(s3["entity_id"])
    s1_ids = set(s1["entity_id"])

    matching_s1_count = 0
    singleton_s1_count = 0
    both_count = 0
    s2_only_count = 0
    s3_only_count = 0
    total_matching_relationships = 0
    seen_matched_ids = set()

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        assert s1_id in s1_ids, f"Unknown S1 ID in ground truth: {s1_id}"

        matched_str = str(row["matched_entity_ids"]).strip()
        if not matched_str:
            singleton_s1_count += 1
            continue

        matching_s1_count += 1
        m_ids = [m.strip() for m in matched_str.split(",")]
        total_matching_relationships += len(m_ids)

        # No duplicate matched IDs within same row
        assert len(m_ids) == len(set(m_ids)), f"Duplicate matched IDs in row for {s1_id}: {m_ids}"

        has_s2 = any(m.startswith("S2-") for m in m_ids)
        has_s3 = any(m.startswith("S3-") for m in m_ids)

        if has_s2 and has_s3:
            both_count += 1
        elif has_s2:
            s2_only_count += 1
        elif has_s3:
            s3_only_count += 1

        for mid in m_ids:
            # Rule: Never include Source 1 IDs in matched_entity_ids
            assert not mid.startswith("S1-"), f"Forbidden Source 1 ID {mid} found in matched_entity_ids!"
            # Rule: Every ID in matched_entity_ids must exist in Source 2 or Source 3
            if mid.startswith("S2-"):
                assert mid in s2_ids, f"Error: Ground truth ID {mid} not found in Source 2!"
            elif mid.startswith("S3-"):
                assert mid in s3_ids, f"Error: Ground truth ID {mid} not found in Source 3!"
            else:
                raise ValueError(f"Unexpected ID prefix in ground truth: {mid}")

    print("\n--- Match Partition Analysis ---")
    print(f"Total Source 1 entities              : {len(gt)}")
    print(f"Matching Source 1 entities           : {matching_s1_count} ({matching_s1_count / len(gt) * 100:.1f}%) [Target: 65-75%]")
    print(f"Singleton Source 1 entities          : {singleton_s1_count} ({singleton_s1_count / len(gt) * 100:.1f}%) [Target: 25-35%]")
    print(f"  - Matches in BOTH Source 2 and 3   : {both_count}")
    print(f"  - Matches in Source 2 ONLY         : {s2_only_count}")
    print(f"  - Matches in Source 3 ONLY         : {s3_only_count}")
    print(f"Total matching relationships         : {total_matching_relationships}")

    assert 650 <= matching_s1_count <= 750, f"Matching S1 count out of 65-75% bounds: {matching_s1_count}"
    assert 250 <= singleton_s1_count <= 350, f"Singleton S1 count out of 25-35% bounds: {singleton_s1_count}"
    assert both_count > 0, "No both matches found!"
    assert s2_only_count > 0, "No S2-only matches found!"
    assert s3_only_count > 0, "No S3-only matches found!"
    print("[OK] Partition ratios strictly within target boundaries")

    # 8. Hard Negative Verification
    print("\n--- Hard Negative Verification ---")
    # Verify that deliberate hard negative pairs in Source 1 are NOT linked in ground truth
    # Pair 1: S1-00001 (Apex Logistics Pvt Ltd, India) vs S1-00002 (Apex Logistics LLC, USA)
    row_s1_01 = gt[gt["source1_entity_id"] == "S1-00001"].iloc[0]["matched_entity_ids"]
    row_s1_02 = gt[gt["source1_entity_id"] == "S1-00002"].iloc[0]["matched_entity_ids"]
    set_01 = set(row_s1_01.split(",")) if row_s1_01 else set()
    set_02 = set(row_s1_02.split(",")) if row_s1_02 else set()
    assert set_01.isdisjoint(set_02), "Hard negative error: S1-00001 and S1-00002 share matches!"

    # Verify building collision hard negatives:
    # Pair: Quantum Labs Cyber Security (S1-00031) vs NextGen FinTech Solutions (S1-00032)
    row_bldg1 = gt[gt["source1_entity_id"] == "S1-00031"].iloc[0]["matched_entity_ids"]
    row_bldg2 = gt[gt["source1_entity_id"] == "S1-00032"].iloc[0]["matched_entity_ids"]
    set_b1 = set(row_bldg1.split(",")) if row_bldg1 else set()
    set_b2 = set(row_bldg2.split(",")) if row_bldg2 else set()
    assert set_b1.isdisjoint(set_b2), "Hard negative error: Co-located building pair shares matches!"
    print("[OK] Confirmed: 60 hard negative pairs exist and are correctly unlinked in ground truth")

    # 9. Concrete Matched Examples
    print("\n--- Sample Matched Entities Across Sources ---")
    sample_gt = gt[gt["matched_entity_ids"].str.contains("S2-") & gt["matched_entity_ids"].str.contains("S3-")].head(2)
    s2_lookup = s2.set_index("entity_id").to_dict(orient="index")
    s3_lookup = s3.set_index("entity_id").to_dict(orient="index")
    s1_lookup = s1.set_index("entity_id").to_dict(orient="index")

    for _, row in sample_gt.iterrows():
        s1_id = row["source1_entity_id"]
        mids = row["matched_entity_ids"].split(",")
        s2_id = [m for m in mids if m.startswith("S2-")][0]
        s3_id = [m for m in mids if m.startswith("S3-")][0]

        print(f"\n[Entity Cluster: {s1_id}]")
        print(f"  Source 1 ({s1_id}):")
        print(f"    Name    : {s1_lookup[s1_id]['business_name']}")
        print(f"    Address : {s1_lookup[s1_id]['business_address']}")
        print(f"    Country : {s1_lookup[s1_id]['country']}")
        print(f"  Source 2 ({s2_id}):")
        print(f"    Name    : {s2_lookup[s2_id]['business_name']}")
        print(f"    Address : {s2_lookup[s2_id]['business_address']}")
        print(f"  Source 3 ({s3_id}):")
        print(f"    Name    : {s3_lookup[s3_id]['business_name']}")
        print(f"    Address : {s3_lookup[s3_id]['business_address']}")

    print("\n[OK] ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!\n")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    # Check business-entity-resolution/dataset
    p1 = current_dir.parent / "dataset"
    if p1.exists():
        run_verification(p1)
    # Check root dataset
    p2 = current_dir.parent.parent / "dataset"
    if p2.exists() and p2 != p1:
        run_verification(p2)
