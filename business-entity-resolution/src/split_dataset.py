"""
Train / Test Dataset Splitter for Business Entity Resolution (Step 2)
====================================================================

Purpose:
--------
Splits the synthetic dataset into 80% Training and 20% Testing partitions
strictly at the Source 1 ENTITY level, ensuring 100% preservation of
entity clusters and zero data leakage.

Entity Cluster Preservation Rule:
---------------------------------
If S1-xxxxx is assigned to TRAIN:
  -> All linked S2 and S3 records via ground_truth.tsv MUST be in TRAIN.
If S1-yyyyy is assigned to TEST:
  -> All linked S2 and S3 records via ground_truth.tsv MUST be in TEST.

Unlinked / Distractor Records:
------------------------------
Distractor records with no ground truth match are deterministically split 80/20
using random seed 42, maintaining realistic noise in both partitions.

Master Dataset Immutability:
----------------------------
The original four master files:
- dataset/source1.tsv
- dataset/source2.tsv
- dataset/source3.tsv
- dataset/ground_truth.tsv
remain strictly unchanged.
"""

import hashlib
import random
from pathlib import Path
import pandas as pd

RANDOM_SEED = 42


def compute_file_hash(filepath: Path) -> str:
    """Computes SHA256 hash of a file for immutability verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def split_and_validate(dataset_dir: Path):
    """
    Executes the 80/20 entity cluster split on dataset_dir and validates
    all partition, integrity, and anti-leakage constraints.
    """
    print(f"\n========================================")
    print(f"PROCESSING DATASET AT: {dataset_dir}")
    print(f"========================================")

    s1_path = dataset_dir / "source1.tsv"
    s2_path = dataset_dir / "source2.tsv"
    s3_path = dataset_dir / "source3.tsv"
    gt_path = dataset_dir / "ground_truth.tsv"

    # Pre-split hashes to verify immutability
    initial_hashes = {
        "s1": compute_file_hash(s1_path),
        "s2": compute_file_hash(s2_path),
        "s3": compute_file_hash(s3_path),
        "gt": compute_file_hash(gt_path),
    }

    # Load master TSV files with sep="\t"
    s1 = pd.read_csv(s1_path, sep="\t", keep_default_na=False)
    s2 = pd.read_csv(s2_path, sep="\t", keep_default_na=False)
    s3 = pd.read_csv(s3_path, sep="\t", keep_default_na=False)
    gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)

    total_s1 = len(s1)
    assert total_s1 == 1000, f"Expected 1,000 Source 1 records, got {total_s1}"

    # Deterministic split of Source 1 entities (80% train / 20% test)
    rng = random.Random(RANDOM_SEED)
    s1_id_list = list(s1["entity_id"])
    rng.shuffle(s1_id_list)

    train_size = int(0.80 * total_s1)  # 800
    test_size = total_s1 - train_size  # 200

    train_s1_set = set(s1_id_list[:train_size])
    test_s1_set = set(s1_id_list[train_size:])

    assert len(train_s1_set) == 800
    assert len(test_s1_set) == 200
    assert train_s1_set.isdisjoint(test_s1_set)

    # Map entity clusters from ground truth
    train_s2_linked = set()
    train_s3_linked = set()
    test_s2_linked = set()
    test_s3_linked = set()

    gt_train_rows = []
    gt_test_rows = []

    train_matching_s1 = 0
    train_singleton_s1 = 0
    test_matching_s1 = 0
    test_singleton_s1 = 0
    train_rel_count = 0
    test_rel_count = 0

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]).strip()
        matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]

        if s1_id in train_s1_set:
            gt_train_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched_ids),
            })
            if matched_ids:
                train_matching_s1 += 1
                train_rel_count += len(matched_ids)
                for mid in matched_ids:
                    if mid.startswith("S2-"):
                        train_s2_linked.add(mid)
                    elif mid.startswith("S3-"):
                        train_s3_linked.add(mid)
            else:
                train_singleton_s1 += 1

        elif s1_id in test_s1_set:
            gt_test_rows.append({
                "source1_entity_id": s1_id,
                "matched_entity_ids": ",".join(matched_ids),
            })
            if matched_ids:
                test_matching_s1 += 1
                test_rel_count += len(matched_ids)
                for mid in matched_ids:
                    if mid.startswith("S2-"):
                        test_s2_linked.add(mid)
                    elif mid.startswith("S3-"):
                        test_s3_linked.add(mid)
            else:
                test_singleton_s1 += 1
        else:
            raise ValueError(f"Uncategorized Source 1 ID: {s1_id}")

    # Handle unlinked / distractor records in Source 2 and Source 3
    all_s2_linked = train_s2_linked.union(test_s2_linked)
    all_s3_linked = train_s3_linked.union(test_s3_linked)

    distractor_s2_list = [eid for eid in s2["entity_id"] if eid not in all_s2_linked]
    distractor_s3_list = [eid for eid in s3["entity_id"] if eid not in all_s3_linked]

    # Deterministically split distractors 80% train / 20% test
    rng.shuffle(distractor_s2_list)
    s2_dist_train_count = int(0.80 * len(distractor_s2_list))
    train_s2_distractors = set(distractor_s2_list[:s2_dist_train_count])
    test_s2_distractors = set(distractor_s2_list[s2_dist_train_count:])

    rng.shuffle(distractor_s3_list)
    s3_dist_train_count = int(0.80 * len(distractor_s3_list))
    train_s3_distractors = set(distractor_s3_list[:s3_dist_train_count])
    test_s3_distractors = set(distractor_s3_list[s3_dist_train_count:])

    # Complete target sets for Source 2 and Source 3
    train_s2_all_ids = train_s2_linked.union(train_s2_distractors)
    test_s2_all_ids = test_s2_linked.union(test_s2_distractors)

    train_s3_all_ids = train_s3_linked.union(train_s3_distractors)
    test_s3_all_ids = test_s3_linked.union(test_s3_distractors)

    # Build DataFrames
    s1_train = s1[s1["entity_id"].isin(train_s1_set)].copy()
    s1_test = s1[s1["entity_id"].isin(test_s1_set)].copy()

    s2_train = s2[s2["entity_id"].isin(train_s2_all_ids)].copy()
    s2_test = s2[s2["entity_id"].isin(test_s2_all_ids)].copy()

    s3_train = s3[s3["entity_id"].isin(train_s3_all_ids)].copy()
    s3_test = s3[s3["entity_id"].isin(test_s3_all_ids)].copy()

    gt_train = pd.DataFrame(gt_train_rows)
    gt_test = pd.DataFrame(gt_test_rows)

    # Output directories
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Write train/test TSV files
    s1_train.to_csv(train_dir / "source1_train.tsv", sep="\t", index=False)
    s2_train.to_csv(train_dir / "source2_train.tsv", sep="\t", index=False)
    s3_train.to_csv(train_dir / "source3_train.tsv", sep="\t", index=False)
    gt_train.to_csv(train_dir / "ground_truth_train.tsv", sep="\t", index=False)

    s1_test.to_csv(test_dir / "source1_test.tsv", sep="\t", index=False)
    s2_test.to_csv(test_dir / "source2_test.tsv", sep="\t", index=False)
    s3_test.to_csv(test_dir / "source3_test.tsv", sep="\t", index=False)
    gt_test.to_csv(test_dir / "ground_truth_test.tsv", sep="\t", index=False)

    # =================================================================
    # VALIDATION CHECKS
    # =================================================================

    # A & B: Source 1 partition check
    assert len(s1_train) == 800, f"Expected 800 s1_train, got {len(s1_train)}"
    assert len(s1_test) == 200, f"Expected 200 s1_test, got {len(s1_test)}"
    assert set(s1_train["entity_id"]).isdisjoint(set(s1_test["entity_id"])), "Leak: S1 ID in both train and test!"
    assert set(s1_train["entity_id"]).union(set(s1_test["entity_id"])) == set(s1["entity_id"]), "Missing S1 IDs!"

    # C & E: Entity cluster integrity and ground truth target existence
    s2_train_id_set = set(s2_train["entity_id"])
    s3_train_id_set = set(s3_train["entity_id"])
    s2_test_id_set = set(s2_test["entity_id"])
    s3_test_id_set = set(s3_test["entity_id"])

    for _, row in gt_train.iterrows():
        s1_id = row["source1_entity_id"]
        assert s1_id in train_s1_set, f"S1 ID {s1_id} in gt_train but not s1_train!"
        matched = str(row["matched_entity_ids"]).strip()
        if matched:
            for mid in matched.split(","):
                mid = mid.strip()
                assert not mid.startswith("S1-"), f"S1 ID found in matched_entity_ids: {mid}"
                if mid.startswith("S2-"):
                    assert mid in s2_train_id_set, f"Ground truth ID {mid} not in source2_train!"
                    assert mid not in s2_test_id_set, f"Leak: Train matched S2 ID {mid} found in source2_test!"
                elif mid.startswith("S3-"):
                    assert mid in s3_train_id_set, f"Ground truth ID {mid} not in source3_train!"
                    assert mid not in s3_test_id_set, f"Leak: Train matched S3 ID {mid} found in source3_test!"

    for _, row in gt_test.iterrows():
        s1_id = row["source1_entity_id"]
        assert s1_id in test_s1_set, f"S1 ID {s1_id} in gt_test but not s1_test!"
        matched = str(row["matched_entity_ids"]).strip()
        if matched:
            for mid in matched.split(","):
                mid = mid.strip()
                assert not mid.startswith("S1-"), f"S1 ID found in matched_entity_ids: {mid}"
                if mid.startswith("S2-"):
                    assert mid in s2_test_id_set, f"Ground truth ID {mid} not in source2_test!"
                    assert mid not in s2_train_id_set, f"Leak: Test matched S2 ID {mid} found in source2_train!"
                elif mid.startswith("S3-"):
                    assert mid in s3_test_id_set, f"Ground truth ID {mid} not in source3_test!"
                    assert mid not in s3_train_id_set, f"Leak: Test matched S3 ID {mid} found in source3_train!"

    # D: Duplicate ID check
    assert s1_train["entity_id"].is_unique, "Duplicate IDs in source1_train!"
    assert s1_test["entity_id"].is_unique, "Duplicate IDs in source1_test!"
    assert s2_train["entity_id"].is_unique, "Duplicate IDs in source2_train!"
    assert s2_test["entity_id"].is_unique, "Duplicate IDs in source2_test!"
    assert s3_train["entity_id"].is_unique, "Duplicate IDs in source3_train!"
    assert s3_test["entity_id"].is_unique, "Duplicate IDs in source3_test!"
    assert gt_train["source1_entity_id"].is_unique, "Duplicate source1_entity_id in gt_train!"
    assert gt_test["source1_entity_id"].is_unique, "Duplicate source1_entity_id in gt_test!"

    # Explicit Data Leakage Check
    assert train_s2_linked.isdisjoint(test_s2_linked), "Leakage: Overlapping linked S2 records between train & test!"
    assert train_s3_linked.isdisjoint(test_s3_linked), "Leakage: Overlapping linked S3 records between train & test!"
    assert train_s2_all_ids.isdisjoint(test_s2_all_ids), "Leakage: Overlapping S2 records between train & test!"
    assert train_s3_all_ids.isdisjoint(test_s3_all_ids), "Leakage: Overlapping S3 records between train & test!"

    # G: Master datasets remain unchanged
    post_hashes = {
        "s1": compute_file_hash(s1_path),
        "s2": compute_file_hash(s2_path),
        "s3": compute_file_hash(s3_path),
        "gt": compute_file_hash(gt_path),
    }
    for key, initial_hash in initial_hashes.items():
        assert initial_hash == post_hashes[key], f"Master file {key} was modified!"

    # Print required summary format
    print("\n========================================")
    print("TRAIN / TEST SPLIT SUMMARY")
    print("==========================")
    print("\nTRAIN")
    print(f"Source 1 records: {len(s1_train)}")
    print(f"Source 2 records: {len(s2_train)}")
    print(f"Source 3 records: {len(s3_train)}")
    print(f"Matching relationships: {train_rel_count}")
    print(f"Singleton Source 1 entities: {train_singleton_s1}")

    print("\nTEST")
    print(f"Source 1 records: {len(s1_test)}")
    print(f"Source 2 records: {len(s2_test)}")
    print(f"Source 3 records: {len(s3_test)}")
    print(f"Matching relationships: {test_rel_count}")
    print(f"Singleton Source 1 entities: {test_singleton_s1}")

    print(f"\nSource 1 assigned to Train : {len(s1_train) / total_s1 * 100:.1f}%")
    print(f"Source 1 assigned to Test  : {len(s1_test) / total_s1 * 100:.1f}%")

    print("\n========================================")
    print("VALIDATION")
    print("==========")
    print("Source 1 partition check: PASSED")
    print("Ground truth integrity: PASSED")
    print("Entity cluster integrity: PASSED")
    print("Duplicate ID check: PASSED")
    print("DATA LEAKAGE CHECK: PASSED")
    print("Master datasets unchanged: PASSED")
    print("========================================\n")


def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent  # business-entity-resolution

    target_dirs = [
        project_root / "dataset",
    ]

    workspace_root = project_root.parent
    if (workspace_root / "dataset") != (project_root / "dataset"):
        target_dirs.append(workspace_root / "dataset")

    for d in target_dirs:
        if d.exists():
            split_and_validate(d)


if __name__ == "__main__":
    main()
