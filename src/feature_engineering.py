"""
Feature Engineering Module for Business Entity Resolution
=========================================================
Computes deterministic string similarity, token overlap, exact match,
and structural features between candidate business records.

Features are computed using normalized text representations to ensure
invariance against noise, casing, formatting, and legal suffix variations.

Similarity scale:
- All string similarity scores (RapidFuzz ratios and token overlap percentages)
  are measured on a consistent 0.0 to 100.0 scale.
- Match flags (country_match, name_exact_match, address_exact_match) are binary (0 or 1).
- Length differences are non-negative absolute integer character differences.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from rapidfuzz import fuzz

try:
    import sklearn
except ImportError:
    sklearn = None

# Exact 15 ML Feature Columns (excluding any entity IDs, names, or target labels)
FEATURE_COLUMNS = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "address_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_partial_ratio",
    "country_match",
    "name_length_difference",
    "address_length_difference",
    "name_token_overlap",
    "address_token_overlap",
    "name_exact_match",
    "address_exact_match",
]


def compute_token_overlap(str1: str, str2: str) -> float:
    """
    Computes Jaccard token overlap similarity percentage (0.0 to 100.0).

    Formula:
        (|tokens1 ∩ tokens2| / |tokens1 ∪ tokens2|) * 100.0
    Returns 0.0 if both strings contain no tokens.
    """
    tokens1 = set(str1.split()) if str1 else set()
    tokens2 = set(str2.split()) if str2 else set()
    union = tokens1 | tokens2
    if not union:
        return 0.0
    intersection = tokens1 & tokens2
    return float((len(intersection) / len(union)) * 100.0)


def extract_pair_features(
    name1: str,
    addr1: str,
    country1: str,
    name2: str,
    addr2: str,
    country2: str,
) -> Dict[str, Any]:
    """
    Computes the 15 engineered similarity features for a single entity pair.

    All input fields should be the normalized strings.

    Returns:
        dict: Mapping of feature names to numeric values.
    """
    n1 = str(name1) if pd.notna(name1) else ""
    a1 = str(addr1) if pd.notna(addr1) else ""
    c1 = str(country1).strip().lower() if pd.notna(country1) else ""

    n2 = str(name2) if pd.notna(name2) else ""
    a2 = str(addr2) if pd.notna(addr2) else ""
    c2 = str(country2).strip().lower() if pd.notna(country2) else ""

    # A. Business Name Features (0-100 scale)
    name_ratio = float(fuzz.ratio(n1, n2))
    name_token_sort_ratio = float(fuzz.token_sort_ratio(n1, n2))
    name_token_set_ratio = float(fuzz.token_set_ratio(n1, n2))
    name_partial_ratio = float(fuzz.partial_ratio(n1, n2))

    # B. Address Features (0-100 scale)
    address_ratio = float(fuzz.ratio(a1, a2))
    address_token_sort_ratio = float(fuzz.token_sort_ratio(a1, a2))
    address_token_set_ratio = float(fuzz.token_set_ratio(a1, a2))
    address_partial_ratio = float(fuzz.partial_ratio(a1, a2))

    # C. Country Feature (Binary 0 or 1)
    country_match = 1 if (c1 and c2 and c1 == c2) else 0

    # D. Text Length Differences (Absolute character count differences)
    name_length_diff = abs(len(n1) - len(n2))
    address_length_diff = abs(len(a1) - len(a2))

    # E. Token Overlap Features (0-100 scale)
    name_token_overlap = compute_token_overlap(n1, n2)
    address_token_overlap = compute_token_overlap(a1, a2)

    # F. Exact Match Features (Binary 0 or 1)
    name_exact_match = 1 if (n1 and n2 and n1 == n2) else 0
    address_exact_match = 1 if (a1 and a2 and a1 == a2) else 0

    return {
        "name_ratio": name_ratio,
        "name_token_sort_ratio": name_token_sort_ratio,
        "name_token_set_ratio": name_token_set_ratio,
        "name_partial_ratio": name_partial_ratio,
        "address_ratio": address_ratio,
        "address_token_sort_ratio": address_token_sort_ratio,
        "address_token_set_ratio": address_token_set_ratio,
        "address_partial_ratio": address_partial_ratio,
        "country_match": country_match,
        "name_length_difference": name_length_diff,
        "address_length_difference": address_length_diff,
        "name_token_overlap": name_token_overlap,
        "address_token_overlap": address_token_overlap,
        "name_exact_match": name_exact_match,
        "address_exact_match": address_exact_match,
    }


def compute_features_for_pairs(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Computes feature matrix for a dataframe containing candidate pairs.

    Expected columns in pairs_df:
        - source1_entity_id
        - source_entity_id
        - source_type ('source2' or 'source3')
        - label (optional, 0 or 1)

    Returns:
        pd.DataFrame: Engineered feature matrix containing only the 15 features
                      plus 'label' (if present in pairs_df).
    """
    # Create lookup dicts for fast indexed access
    s1_dict = s1_df.set_index("entity_id")[
        ["business_name_normalized", "business_address_normalized", "country_normalized"]
    ].to_dict("index")

    s2_dict = s2_df.set_index("entity_id")[
        ["business_name_normalized", "business_address_normalized", "country_normalized"]
    ].to_dict("index")

    s3_dict = s3_df.set_index("entity_id")[
        ["business_name_normalized", "business_address_normalized", "country_normalized"]
    ].to_dict("index")

    feature_records: List[Dict[str, Any]] = []

    has_label = "label" in pairs_df.columns
    records = pairs_df.to_dict("records")

    for row in records:
        s1_id = row["source1_entity_id"]
        cand_id = row["source_entity_id"]
        s_type = row["source_type"]

        s1_row = s1_dict.get(s1_id, {})
        cand_dict = s2_dict if s_type == "source2" else s3_dict
        cand_row = cand_dict.get(cand_id, {})

        feats = extract_pair_features(
            name1=s1_row.get("business_name_normalized", ""),
            addr1=s1_row.get("business_address_normalized", ""),
            country1=s1_row.get("country_normalized", ""),
            name2=cand_row.get("business_name_normalized", ""),
            addr2=cand_row.get("business_address_normalized", ""),
            country2=cand_row.get("country_normalized", ""),
        )

        if has_label:
            feats["label"] = int(row["label"])

        feature_records.append(feats)

    cols = list(FEATURE_COLUMNS)
    if has_label:
        cols.append("label")

    return pd.DataFrame(feature_records)[cols]
