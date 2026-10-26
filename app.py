"""
Business Entity Resolution Web Application
===========================================
Flask-based deployment of the trained Logistic Regression model
for cross-source business entity resolution.

Routes:
  GET /                         -> Main interactive dashboard
  GET /api/entities             -> List of all 200 Source 1 test businesses
  GET /api/match/<s1_id>        -> Match prediction for selected Source 1 entity
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib
from rapidfuzz import fuzz
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Path resolution (supports root and subfolder execution)
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent

def find_file(relative_paths: List[str]) -> Path:
    search_roots = [
        CURRENT_DIR,
        CURRENT_DIR / "business-entity-resolution",
        CURRENT_DIR.parent / "business-entity-resolution"
    ]
    for rel_path in relative_paths:
        for root in search_roots:
            candidate = root / rel_path
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not locate any of: {relative_paths}")

MODEL_PATH = find_file(["models/logistic_regression_model.joblib"])

S1_PATH = find_file([
    "dataset/preprocessed/test/source1_test_preprocessed.tsv",
    "dataset/preprocessed/test/source1_test.tsv"
])
S2_PATH = find_file([
    "dataset/preprocessed/test/source2_test_preprocessed.tsv",
    "dataset/preprocessed/test/source2_test.tsv"
])
S3_PATH = find_file([
    "dataset/preprocessed/test/source3_test_preprocessed.tsv",
    "dataset/preprocessed/test/source3_test.tsv"
])

# ---------------------------------------------------------------------------
# Load model & datasets at startup
# ---------------------------------------------------------------------------
print("Loading model and datasets ...")
model = joblib.load(MODEL_PATH)

s1_df = pd.read_csv(S1_PATH, sep="\t")
s2_df = pd.read_csv(S2_PATH, sep="\t")
s3_df = pd.read_csv(S3_PATH, sep="\t")

print(f"Loaded model from: {MODEL_PATH}")
print(f"Loaded {len(s1_df)} Source 1 test entities")
print(f"Loaded {len(s2_df)} Source 2 test entities")
print(f"Loaded {len(s3_df)} Source 3 test entities")

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

THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Feature extraction & token overlap (identical to Step 5, 6, 7)
# ---------------------------------------------------------------------------
def compute_token_overlap(s1: str, s2: str) -> float:
    t1 = set(s1.split()) if s1 else set()
    t2 = set(s2.split()) if s2 else set()
    union = t1 | t2
    if not union:
        return 0.0
    return float(len(t1 & t2) / len(union) * 100.0)


def extract_features(name1: Any, addr1: Any, country1: Any,
                     name2: Any, addr2: Any, country2: Any) -> Dict[str, Any]:
    n1 = str(name1) if pd.notna(name1) else ""
    a1 = str(addr1) if pd.notna(addr1) else ""
    c1 = str(country1).strip().lower() if pd.notna(country1) else ""
    n2 = str(name2) if pd.notna(name2) else ""
    a2 = str(addr2) if pd.notna(addr2) else ""
    c2 = str(country2).strip().lower() if pd.notna(country2) else ""

    return {
        "name_ratio": float(fuzz.ratio(n1, n2)),
        "name_token_sort_ratio": float(fuzz.token_sort_ratio(n1, n2)),
        "name_token_set_ratio": float(fuzz.token_set_ratio(n1, n2)),
        "name_partial_ratio": float(fuzz.partial_ratio(n1, n2)),
        "address_ratio": float(fuzz.ratio(a1, a2)),
        "address_token_sort_ratio": float(fuzz.token_sort_ratio(a1, a2)),
        "address_token_set_ratio": float(fuzz.token_set_ratio(a1, a2)),
        "address_partial_ratio": float(fuzz.partial_ratio(a1, a2)),
        "country_match": 1 if (c1 and c2 and c1 == c2) else 0,
        "name_length_difference": abs(len(n1) - len(n2)),
        "address_length_difference": abs(len(a1) - len(a2)),
        "name_token_overlap": compute_token_overlap(n1, n2),
        "address_token_overlap": compute_token_overlap(a1, a2),
        "name_exact_match": 1 if (n1 and n2 and n1 == n2) else 0,
        "address_exact_match": 1 if (a1 and a2 and a1 == a2) else 0,
    }


# ---------------------------------------------------------------------------
# Match resolution logic for a given Source 1 entity
# ---------------------------------------------------------------------------
def resolve_matches_for_entity(s1_id: str) -> Optional[Dict[str, Any]]:
    s1_rows = s1_df[s1_df["entity_id"] == s1_id]
    if s1_rows.empty:
        return None

    r1 = s1_rows.iloc[0]
    n1 = str(r1["business_name_normalized"])
    a1 = str(r1["business_address_normalized"])
    c1 = str(r1["country_normalized"])

    source1_info = {
        "entity_id": str(r1["entity_id"]),
        "business_name": str(r1["business_name"]),
        "business_address": str(r1["business_address"]),
        "country": str(r1["country"]),
    }

    results = {
        "source1": source1_info,
        "source2_matches": [],
        "source3_matches": []
    }

    for stype, df, out_key in [("source2", s2_df, "source2_matches"),
                               ("source3", s3_df, "source3_matches")]:
        cands = df[df["country_normalized"] == c1]
        candidate_features = []
        candidate_records = []

        for _, r2 in cands.iterrows():
            n2 = str(r2["business_name_normalized"])
            a2 = str(r2["business_address_normalized"])
            n_sim = fuzz.token_sort_ratio(n1, n2)
            n_set = fuzz.token_set_ratio(n1, n2)
            a_sim = fuzz.token_sort_ratio(a1, a2)

            # Candidate generation blocking rule
            if n_sim >= 50 or n_set >= 65 or a_sim >= 50:
                f = extract_features(n1, a1, c1, n2, a2, r2["country_normalized"])
                candidate_features.append([f[col] for col in FEATURE_COLUMNS])
                candidate_records.append(r2)

        if candidate_features:
            X = pd.DataFrame(candidate_features, columns=FEATURE_COLUMNS)
            probas = model.predict_proba(X)[:, 1]

            for r2, prob in zip(candidate_records, probas):
                if prob >= THRESHOLD:
                    prob_val = float(prob)
                    results[out_key].append({
                        "entity_id": str(r2["entity_id"]),
                        "business_name": str(r2["business_name"]),
                        "business_address": str(r2["business_address"]),
                        "country": str(r2["country"]),
                        "match_probability": round(prob_val, 4),
                        "match_probability_pct": f"{prob_val * 100:.1f}%",
                        "prediction": 1
                    })

        # Sort descending by probability
        results[out_key].sort(key=lambda x: x["match_probability"], reverse=True)

    return results


# ---------------------------------------------------------------------------
# Flask App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/entities", methods=["GET"])
def get_entities():
    entities = []
    for _, row in s1_df.iterrows():
        eid = str(row["entity_id"])
        bname = str(row["business_name"])
        country = str(row["country"])
        addr = str(row["business_address"])
        entities.append({
            "entity_id": eid,
            "business_name": bname,
            "country": country,
            "business_address": addr,
            "label": f"{eid} — {bname} — {country}"
        })
    return jsonify(entities)


@app.route("/api/source2", methods=["GET"])
def get_source2():
    entities = []
    for _, row in s2_df.iterrows():
        entities.append({
            "entity_id": str(row["entity_id"]),
            "business_name": str(row["business_name"]),
            "country": str(row["country"]),
            "business_address": str(row["business_address"]),
            "label": f"{row['entity_id']} — {row['business_name']} — {row['country']}"
        })
    return jsonify(entities)


@app.route("/api/source3", methods=["GET"])
def get_source3():
    entities = []
    for _, row in s3_df.iterrows():
        entities.append({
            "entity_id": str(row["entity_id"]),
            "business_name": str(row["business_name"]),
            "country": str(row["country"]),
            "business_address": str(row["business_address"]),
            "label": f"{row['entity_id']} — {row['business_name']} — {row['country']}"
        })
    return jsonify(entities)


@app.route("/api/match/<source1_entity_id>", methods=["GET"])
def match_entity(source1_entity_id: str):
    res = resolve_matches_for_entity(source1_entity_id.strip())
    if res is None:
        return jsonify({"error": f"Entity ID '{source1_entity_id}' not found in Source 1 test dataset."}), 404
    return jsonify(res)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Business Entity Resolution Web App")
    print("Running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=False)
