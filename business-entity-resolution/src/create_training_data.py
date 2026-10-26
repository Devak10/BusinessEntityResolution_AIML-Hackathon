"""
Training Data and Feature Generation Pipeline
=============================================
Step 5: Feature Engineering + Training Pair Creation

Executes:
1. Extraction of true positive pairs from ground_truth_train.tsv.
2. Candidate blocking and negative pair sampling (hard and easy negatives).
3. Elimination of duplicates and validation against positive match labels.
4. Feature engineering for all pairs using normalized fields and RapidFuzz metrics.
5. Export of training_pairs.tsv, training_features.tsv, and feature_description.txt.
6. Rigorous statistical inspection and 8-point validation suite.
"""

import sys
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

# Add project root and src to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

from feature_engineering import (
    FEATURE_COLUMNS,
    extract_pair_features,
    compute_features_for_pairs,
)


def resolve_paths() -> Tuple[Path, Path]:
    """Resolves primary project root and workspace root."""
    base = Path(__file__).resolve().parent.parent
    workspace_root = Path("d:/BusinessEntityResolution_AIML-Hackathon")
    return base, workspace_root


def get_file_sha256(path: Path) -> str:
    """Computes SHA-256 hash of a file for immutability check."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def generate_feature_description() -> str:
    """Generates detailed documentation for all 15 engineered features."""
    return """BUSINESS ENTITY RESOLUTION - FEATURE DICTIONARY
=================================================
This document defines the 15 engineered numeric features used for supervised
machine learning training in the Business Entity Resolution pipeline.

All features are strictly derived from preprocessed, normalized text fields:
- business_name_normalized
- business_address_normalized
- country_normalized

No raw entity IDs, ground-truth match flags, or database identifiers are used
as features, preventing data leakage.

1. name_ratio
   - Meaning: Standard Levenshtein distance ratio between normalized business names.
   - Range: [0.0, 100.0]
   - Usefulness: Measures overall character-level edit distance similarity. Captures minor typos
     and character swaps (e.g., 'Infotech' vs 'Infotehc').

2. name_token_sort_ratio
   - Meaning: Token-sorted Levenshtein ratio between normalized business names.
   - Range: [0.0, 100.0]
   - Usefulness: Invariant to word reordering (e.g., 'Apex Logistics Pvt Ltd' vs
     'Logistics Apex Pvt Ltd'). Critical for handling variations across ERP and directory sources.

3. name_token_set_ratio
   - Meaning: Token-set ratio between normalized business names.
   - Range: [0.0, 100.0]
   - Usefulness: Handles subset/superset naming discrepancies by comparing the intersection
     against remainder tokens (e.g., 'Veritas Tech' vs 'Veritas Tech Platforms Group Inc').

4. name_partial_ratio
   - Meaning: Best matching substring similarity ratio for normalized business names.
   - Range: [0.0, 100.0]
   - Usefulness: Detects when a shorter brand name appears cleanly inside an extended legal title.

5. address_ratio
   - Meaning: Standard Levenshtein distance ratio between normalized business addresses.
   - Range: [0.0, 100.0]
   - Usefulness: Quantifies character-level fidelity of the physical address string.

6. address_token_sort_ratio
   - Meaning: Token-sorted Levenshtein ratio between normalized addresses.
   - Range: [0.0, 100.0]
   - Usefulness: Handles reordered address components (e.g., 'Plot 18, MIDC, Andheri' vs
     'Andheri East, Plot 18, MIDC').

7. address_token_set_ratio
   - Meaning: Token-set ratio between normalized addresses.
   - Range: [0.0, 100.0]
   - Usefulness: Accommodates partial or landmark-supplemented addresses (e.g., 'Near Metro Station'
     added to an otherwise identical address).

8. address_partial_ratio
   - Meaning: Best matching substring ratio for normalized addresses.
   - Range: [0.0, 100.0]
   - Usefulness: Identifies shared street address lines when suite/floor numbers differ or are omitted.

9. country_match
   - Meaning: Binary flag indicating if normalized country strings are identical.
   - Range: {0, 1}
   - Usefulness: Serves as a vital gating feature to resolve brand collisions between different
     international entities sharing identical or similar names (e.g., Mumbai vs Dallas).

10. name_length_difference
    - Meaning: Absolute difference in character length between the two normalized names.
    - Range: [0, inf) (Integer)
    - Usefulness: Large disparities signal dropped legal terms, distinct corporate entities, or distractors.

11. address_length_difference
    - Meaning: Absolute difference in character length between the two normalized addresses.
    - Range: [0, inf) (Integer)
    - Usefulness: Highlights omissions of postal codes, state abbreviations, or street specifications.

12. name_token_overlap
    - Meaning: Jaccard similarity percentage of token sets between normalized names.
    - Range: [0.0, 100.0]
    - Formula: (|tokens1 ∩ tokens2| / |tokens1 ∪ tokens2|) * 100.0
    - Usefulness: Vocabulary-level overlap measure independent of character sequence order.

13. address_token_overlap
    - Meaning: Jaccard similarity percentage of token sets between normalized addresses.
    - Range: [0.0, 100.0]
    - Formula: (|tokens1 ∩ tokens2| / |tokens1 ∪ tokens2|) * 100.0
    - Usefulness: Quantifies the proportion of shared address tokens (numbers, localities, PIN/ZIP codes).

14. name_exact_match
    - Meaning: Binary flag indicating whether normalized business names are character-identical.
    - Range: {0, 1}
    - Usefulness: High-confidence indicator when paired with high address similarity.

15. address_exact_match
    - Meaning: Binary flag indicating whether normalized addresses are character-identical.
    - Range: {0, 1}
    - Usefulness: Strong corroborating evidence for same-facility or co-located entities.

LABEL (Target y):
- label: Binary {0, 1}
  1 = True Match (Same real-world business entity)
  0 = Non-Match (Different business entities, including hard and easy negatives)
"""


def main():
    base_dir, ws_dir = resolve_paths()

    # Verify input paths
    prep_dir = base_dir / "dataset" / "preprocessed" / "train"
    train_dir = base_dir / "dataset" / "train"
    test_dir = base_dir / "dataset" / "test"
    model_data_dir = base_dir / "dataset" / "model_data"
    model_data_dir.mkdir(parents=True, exist_ok=True)

    s1_path = prep_dir / "source1_train_preprocessed.tsv"
    s2_path = prep_dir / "source2_train_preprocessed.tsv"
    s3_path = prep_dir / "source3_train_preprocessed.tsv"
    gt_path = train_dir / "ground_truth_train.tsv"

    # Pre-computation file immutability hashes for original and preprocessed files
    original_hashes = {
        "s1_train": get_file_sha256(s1_path),
        "s2_train": get_file_sha256(s2_path),
        "s3_train": get_file_sha256(s3_path),
        "gt_train": get_file_sha256(gt_path),
    }

    # Record test file hashes to guarantee test partition remains untouched
    test_files = list(test_dir.glob("*.tsv")) + list(
        (base_dir / "dataset" / "preprocessed" / "test").glob("*.tsv")
    )
    test_hashes = {f.name: get_file_sha256(f) for f in test_files}

    # 1. Load preprocessed training datasets
    s1_df = pd.read_csv(s1_path, sep="\t")
    s2_df = pd.read_csv(s2_path, sep="\t")
    s3_df = pd.read_csv(s3_path, sep="\t")
    gt_df = pd.read_csv(gt_path, sep="\t")

    # 2. Extract Positive Training Pairs from Ground Truth
    positive_pairs: List[Dict[str, Any]] = []
    known_positives: Set[Tuple[str, str]] = set()
    s2_pos_count = 0
    s3_pos_count = 0

    for _, row in gt_df.dropna(subset=["matched_entity_ids"]).iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        matched_str = str(row["matched_entity_ids"]).strip()
        for cand_id in matched_str.split(","):
            cand_id = cand_id.strip()
            if not cand_id:
                continue
            known_positives.add((s1_id, cand_id))
            if cand_id.startswith("S2-"):
                stype = "source2"
                s2_pos_count += 1
            elif cand_id.startswith("S3-"):
                stype = "source3"
                s3_pos_count += 1
            else:
                stype = "unknown"

            positive_pairs.append({
                "source1_entity_id": s1_id,
                "source_entity_id": cand_id,
                "source_type": stype,
                "label": 1,
                "pair_type": "positive",
            })

    pos_count = len(positive_pairs)

    # 3. Candidate Blocking and Negative Pair Sampling
    s1_records = s1_df.to_dict("records")
    s2_records = s2_df.to_dict("records")
    s3_records = s3_df.to_dict("records")

    s2_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for r in s2_records:
        s2_by_country.setdefault(r["country_normalized"], []).append(r)

    s3_by_country: Dict[str, List[Dict[str, Any]]] = {}
    for r in s3_records:
        s3_by_country.setdefault(r["country_normalized"], []).append(r)

    s1_hard_candidates: Dict[str, List[Dict[str, Any]]] = {r["entity_id"]: [] for r in s1_records}
    s1_easy_candidates: Dict[str, List[Dict[str, Any]]] = {r["entity_id"]: [] for r in s1_records}

    # Generate candidate pool per Source 1 entity via candidate blocking (blocking by country)
    for r1 in s1_records:
        s1_id = r1["entity_id"]
        name1 = str(r1["business_name_normalized"])
        addr1 = str(r1["business_address_normalized"])
        c1 = str(r1["country_normalized"])

        for stype, c_dict in [("source2", s2_by_country), ("source3", s3_by_country)]:
            cands = c_dict.get(c1, [])
            for r2 in cands:
                cand_id = r2["entity_id"]
                if (s1_id, cand_id) in known_positives:
                    continue

                name2 = str(r2["business_name_normalized"])
                addr2 = str(r2["business_address_normalized"])

                n_sim = fuzz.token_sort_ratio(name1, name2)
                n_set = fuzz.token_set_ratio(name1, name2)
                a_sim = fuzz.token_sort_ratio(addr1, addr2)

                diff_score = 0.65 * max(n_sim, n_set) + 0.35 * a_sim

                entry = {
                    "source1_entity_id": s1_id,
                    "source_entity_id": cand_id,
                    "source_type": stype,
                    "label": 0,
                    "pair_type": "hard_negative",
                    "difficulty_score": diff_score,
                }

                # Hard negative: high name similarity or high address similarity
                if n_sim >= 50 or n_set >= 65 or a_sim >= 50:
                    s1_hard_candidates[s1_id].append(entry)
                # Easy negative: clear mismatch within same country
                elif n_sim < 45 and a_sim < 40 and n_set < 60:
                    entry["pair_type"] = "easy_negative"
                    s1_easy_candidates[s1_id].append(entry)

    rng = np.random.RandomState(42)

    # Stratified Hard Negative Selection (Target: 1,200)
    # Ensure every S1 entity gets at least 1 top hard negative
    selected_hard: List[Dict[str, Any]] = []
    hard_seen: Set[Tuple[str, str]] = set()

    for s1_id in s1_hard_candidates:
        s1_hard_candidates[s1_id].sort(key=lambda x: x["difficulty_score"], reverse=True)
        if s1_hard_candidates[s1_id]:
            best = s1_hard_candidates[s1_id][0]
            selected_hard.append(best)
            hard_seen.add((best["source1_entity_id"], best["source_entity_id"]))

    # Fill remaining hard negatives up to 1,200 using highest difficulty candidates
    remaining_hard: List[Dict[str, Any]] = []
    for s1_id in s1_hard_candidates:
        for cand in s1_hard_candidates[s1_id][1:]:
            pair_key = (cand["source1_entity_id"], cand["source_entity_id"])
            if pair_key not in hard_seen:
                remaining_hard.append(cand)

    remaining_hard.sort(key=lambda x: x["difficulty_score"], reverse=True)
    for cand in remaining_hard:
        if len(selected_hard) >= 1200:
            break
        selected_hard.append(cand)
        hard_seen.add((cand["source1_entity_id"], cand["source_entity_id"]))

    # Stratified Easy Negative Selection (Target: 1,200)
    # Ensure every S1 entity gets at least 1 easy negative
    selected_easy: List[Dict[str, Any]] = []
    easy_seen: Set[Tuple[str, str]] = set()

    for s1_id, cand_list in s1_easy_candidates.items():
        if cand_list:
            idx = rng.randint(0, len(cand_list))
            cand = cand_list[idx]
            pair_key = (cand["source1_entity_id"], cand["source_entity_id"])
            if pair_key not in hard_seen:
                selected_easy.append(cand)
                easy_seen.add(pair_key)

    # Fill remaining easy negatives up to 1,200
    remaining_easy: List[Dict[str, Any]] = []
    for s1_id, cand_list in s1_easy_candidates.items():
        for cand in cand_list:
            pair_key = (cand["source1_entity_id"], cand["source_entity_id"])
            if pair_key not in hard_seen and pair_key not in easy_seen:
                remaining_easy.append(cand)

    rng.shuffle(remaining_easy)
    for cand in remaining_easy:
        if len(selected_easy) >= 1200:
            break
        selected_easy.append(cand)
        easy_seen.add((cand["source1_entity_id"], cand["source_entity_id"]))

    hard_count = len(selected_hard)
    easy_count = len(selected_easy)
    neg_count = hard_count + easy_count
    total_pairs = pos_count + neg_count

    # Combine all training pairs
    all_pairs = positive_pairs + selected_hard + selected_easy
    pairs_df = pd.DataFrame(all_pairs)

    # 4. Generate Similarity Features
    features_df = compute_features_for_pairs(pairs_df, s1_df, s2_df, s3_df)

    # 5. Export TSV and Documentation Files
    pairs_export_cols = ["source1_entity_id", "source_entity_id", "source_type", "label"]
    pairs_export_df = pairs_df[pairs_export_cols]

    pairs_output_path = model_data_dir / "training_pairs.tsv"
    features_output_path = model_data_dir / "training_features.tsv"
    desc_output_path = model_data_dir / "feature_description.txt"

    pairs_export_df.to_csv(pairs_output_path, sep="\t", index=False)
    features_df.to_csv(features_output_path, sep="\t", index=False)

    with open(desc_output_path, "w", encoding="utf-8") as f:
        f.write(generate_feature_description())

    # Synchronize to workspace root if running from submodule or root
    for target_dir in [ws_dir / "dataset" / "model_data", base_dir / "dataset" / "model_data"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        pairs_export_df.to_csv(target_dir / "training_pairs.tsv", sep="\t", index=False)
        features_df.to_csv(target_dir / "training_features.tsv", sep="\t", index=False)
        with open(target_dir / "feature_description.txt", "w", encoding="utf-8") as f:
            f.write(generate_feature_description())

    # 6. Quality and Leakage Validation
    label_check = set(pairs_df["label"].unique()) == {0, 1}
    pos_integrity = (pairs_df["label"] == 1).sum() == pos_count
    neg_integrity = (pairs_df["label"] == 0).sum() == neg_count

    # Check for duplicate pairs
    pair_tuples = list(zip(pairs_df["source1_entity_id"], pairs_df["source_entity_id"]))
    no_duplicate_pairs = len(pair_tuples) == len(set(pair_tuples))

    # Check zero overlap between negative pairs and ground truth
    neg_pairs_set = set(
        zip(
            pairs_df[pairs_df["label"] == 0]["source1_entity_id"],
            pairs_df[pairs_df["label"] == 0]["source_entity_id"],
        )
    )
    no_label_leakage = len(neg_pairs_set & known_positives) == 0

    # No NaN or Infinite values
    no_nan_values = not features_df.isna().any().any()
    no_inf_values = not np.isinf(features_df.to_numpy()).any()

    # Feature range check
    range_check = True
    for col in FEATURE_COLUMNS:
        c_min = features_df[col].min()
        c_max = features_df[col].max()
        if "ratio" in col or "overlap" in col:
            if c_min < 0.0 or c_max > 100.0:
                range_check = False
        elif col in ["country_match", "name_exact_match", "address_exact_match"]:
            if set(features_df[col].unique()).issubset({0, 1}) is False:
                range_check = False
        elif "difference" in col:
            if c_min < 0:
                range_check = False

    # Test data untouched verification
    test_untouched = True
    for f in test_files:
        if get_file_sha256(f) != test_hashes[f.name]:
            test_untouched = False

    # 7. Print Required Outputs
    pos_pct = (pos_count / total_pairs) * 100
    neg_pct = (neg_count / total_pairs) * 100

    print("========================================")
    print("TRAINING PAIR GENERATION")
    print("========================")
    print()
    print(f"Positive pairs: {pos_count}")
    print(f"Negative pairs: {neg_count}")
    print(f"Total pairs: {total_pairs}")
    print()
    print(f"Positive percentage: {pos_pct:.2f}%")
    print(f"Negative percentage: {neg_pct:.2f}%")
    print()
    print(f"Source 2 positive pairs: {s2_pos_count}")
    print(f"Source 3 positive pairs: {s3_pos_count}")
    print()
    print(f"Hard negative pairs: {hard_count}")
    print(f"Easy negative pairs: {easy_count}")
    print()
    print("========================================")
    print("FEATURE SUMMARY")
    print("===============")
    print()
    print(f"Number of ML features: {len(FEATURE_COLUMNS)}")
    print(f"Feature names: {', '.join(FEATURE_COLUMNS)}")
    print()
    print("========================================")
    print("FEATURE STATISTICAL SUMMARY (MIN / MAX / MEAN)")
    print("========================================")
    stats_df = pd.DataFrame({
        "Feature": FEATURE_COLUMNS,
        "Min": [features_df[c].min() for c in FEATURE_COLUMNS],
        "Max": [features_df[c].max() for c in FEATURE_COLUMNS],
        "Mean": [features_df[c].mean() for c in FEATURE_COLUMNS],
    })
    print(stats_df.to_string(index=False))
    print()

    # 8. Print Sample Pairs (Section 13)
    print("================================================================================")
    print("SAMPLE TRAINING PAIRS DEMONSTRATION")
    print("================================================================================")

    # Build quick lookup for raw names
    s1_name_map = s1_df.set_index("entity_id")["business_name_normalized"].to_dict()
    s2_name_map = s2_df.set_index("entity_id")["business_name_normalized"].to_dict()
    s3_name_map = s3_df.set_index("entity_id")["business_name_normalized"].to_dict()

    # Pick representative samples
    samples_idx = [
        0, 1, 2, 3,  # Positives
        pos_count, pos_count + 1, pos_count + 2, pos_count + 3,  # Hard Negatives
        pos_count + hard_count, pos_count + hard_count + 1, pos_count + hard_count + 2, pos_count + hard_count + 3  # Easy Negatives
    ]

    for idx in samples_idx:
        p_row = pairs_df.iloc[idx]
        f_row = features_df.iloc[idx]
        s1_id = p_row["source1_entity_id"]
        cand_id = p_row["source_entity_id"]
        stype = p_row["source_type"]
        ptype = p_row["pair_type"]
        lbl = p_row["label"]

        n1 = s1_name_map.get(s1_id, "")
        n2 = s2_name_map.get(cand_id, "") if stype == "source2" else s3_name_map.get(cand_id, "")

        print(f"[{ptype.upper()} PAIR - {s1_id} <-> {cand_id}]")
        print(f"  Source 1 business name : {n1}")
        print(f"  Candidate business name: {n2}")
        print(f"  Source type            : {stype}")
        print(f"  Label                  : {lbl}")
        print(f"  Name similarity        : {f_row['name_token_sort_ratio']:.1f}")
        print(f"  Address similarity     : {f_row['address_token_sort_ratio']:.1f}")
        print(f"  Country match          : {int(f_row['country_match'])}")
        print("-" * 80)

    print()
    print("========================================")
    print("TRAINING DATA VALIDATION")
    print("========================")
    print()
    print(f"Positive/negative label check: {'PASSED' if label_check else 'FAILED'}")
    print(f"Positive pair integrity: {'PASSED' if pos_integrity else 'FAILED'}")
    print(f"Negative pair integrity: {'PASSED' if neg_integrity else 'FAILED'}")
    print(f"No label leakage: {'PASSED' if no_label_leakage else 'FAILED'}")
    print(f"No NaN values: {'PASSED' if no_nan_values else 'FAILED'}")
    print(f"No infinite values: {'PASSED' if no_inf_values else 'FAILED'}")
    print(f"Feature range check: {'PASSED' if range_check else 'FAILED'}")
    print(f"Test data untouched: {'PASSED' if test_untouched else 'FAILED'}")
    print()
    print("========================================")


if __name__ == "__main__":
    main()
