"""
Machine Learning Model Training Pipeline
=========================================
Step 6: Train, Evaluate, and Select the Best Model

Trains two models on the engineered feature matrix:
  1. Logistic Regression (with StandardScaler pipeline)
  2. Random Forest Classifier

Evaluation uses an internal 80/20 stratified validation split.
The official test partition is NOT loaded or used at any point.

Metrics emphasised: Precision, F0.5, Recall
F0.5 formula: (1 + 0.5^2) * P * R / (0.5^2 * P + R)

Similarity scale: 0-100 (as produced by feature_engineering.py)
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = Path("d:/BusinessEntityResolution_AIML-Hackathon")

FEATURES_PATH = BASE_DIR / "dataset" / "model_data" / "training_features.tsv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "model_results"
MODEL_DATA_DIR = BASE_DIR / "dataset" / "model_data"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mirror directories in workspace root
for d in [
    WORKSPACE_ROOT / "models",
    WORKSPACE_ROOT / "model_results",
    WORKSPACE_ROOT / "dataset" / "model_data",
]:
    d.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# Helper: F0.5 score
# ---------------------------------------------------------------------------
def f_beta_score(precision: float, recall: float, beta: float = 0.5) -> float:
    """
    F-beta score.
      F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
    Returns 0.0 if denominator is zero.
    """
    beta_sq = beta ** 2
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denom


# ---------------------------------------------------------------------------
# 1. DATA LEAKAGE CHECK
# ---------------------------------------------------------------------------
def data_leakage_check():
    """Confirm no test data files are loaded during training."""
    test_paths = [
        BASE_DIR / "dataset" / "test",
        BASE_DIR / "dataset" / "preprocessed" / "test",
        WORKSPACE_ROOT / "dataset" / "test",
    ]
    # No test path should be imported or referenced — assert FEATURES_PATH is training only
    assert "test" not in str(FEATURES_PATH).lower(), "Test path detected in features file path!"
    assert FEATURES_PATH.exists(), f"training_features.tsv not found: {FEATURES_PATH}"
    print("DATA LEAKAGE CHECK: PASSED")
    print()


# ---------------------------------------------------------------------------
# 2. LOAD FEATURES
# ---------------------------------------------------------------------------
def load_features():
    df = pd.read_csv(FEATURES_PATH, sep="\t")
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()
    return X, y, df


# ---------------------------------------------------------------------------
# 3. MODEL SANITY CHECK
# ---------------------------------------------------------------------------
def sanity_check(X: pd.DataFrame, y: pd.Series):
    assert X.shape[1] == 15, f"Expected 15 features, got {X.shape[1]}"
    assert set(y.unique()).issubset({0, 1}), f"Labels not binary: {y.unique()}"
    assert not X.isna().any().any(), "NaN values found in X!"
    assert not np.isinf(X.to_numpy()).any(), "Infinite values found in X!"
    print("MODEL SANITY CHECK: PASSED")
    print()


# ---------------------------------------------------------------------------
# 4. TRAIN / VALIDATION SPLIT
# ---------------------------------------------------------------------------
def make_split(X: pd.DataFrame, y: pd.Series):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_val, y_train, y_val


# ---------------------------------------------------------------------------
# 5. EVALUATE MODEL
# ---------------------------------------------------------------------------
def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series, threshold: float = 0.50):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f05 = f_beta_score(prec, rec, beta=0.5)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    acc = accuracy_score(y_val, y_pred)

    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
        "precision": prec,
        "recall": rec,
        "f0_5": f05,
        "f1": f1,
        "accuracy": acc,
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# ---------------------------------------------------------------------------
# 6. PRINT CONFUSION MATRIX WITH EXPLANATION
# ---------------------------------------------------------------------------
def print_confusion_matrix(name: str, metrics: dict):
    print(f"  Confusion Matrix -- {name}")
    print(f"  +------------------------------------------+")
    print(f"  |             Predicted                    |")
    print(f"  |         Non-Match    Match               |")
    print(f"  | Actual Non-Match TN={metrics['TN']:4d}   FP={metrics['FP']:4d}  |")
    print(f"  | Actual Match     FN={metrics['FN']:4d}   TP={metrics['TP']:4d}  |")
    print(f"  +------------------------------------------+")
    print()
    print(f"  Interpretation:")
    print(f"    True Positive  (TP={metrics['TP']:4d}): Correctly identified matching pairs.")
    print(f"    True Negative  (TN={metrics['TN']:4d}): Correctly identified non-matching pairs.")
    print(f"    False Positive (FP={metrics['FP']:4d}): Model INCORRECTLY said two different")
    print(f"                              businesses are the same. (Bad for precision!)")
    print(f"    False Negative (FN={metrics['FN']:4d}): Model INCORRECTLY said two matching")
    print(f"                              businesses are different. (Bad for recall!)")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    # 1. Data leakage check
    data_leakage_check()

    # 2. Load features
    X, y, full_df = load_features()

    # 3. Sanity check
    sanity_check(X, y)

    # 4. Split
    X_train, X_val, y_train, y_val = make_split(X, y)

    print("========================================")
    print("TRAINING / VALIDATION SPLIT")
    print("========================================")
    print(f"Training pairs  : {len(X_train)}")
    print(f"Validation pairs: {len(X_val)}")
    print()
    print(f"Training positive : {int(y_train.sum())}  ({y_train.mean()*100:.1f}%)")
    print(f"Training negative : {int((y_train==0).sum())}  ({(1-y_train.mean())*100:.1f}%)")
    print()
    print(f"Validation positive: {int(y_val.sum())}  ({y_val.mean()*100:.1f}%)")
    print(f"Validation negative: {int((y_val==0).sum())}  ({(1-y_val.mean())*100:.1f}%)")
    print()

    # -------------------------------------------------------------------------
    # 5. MODEL 1 — LOGISTIC REGRESSION (Pipeline: StandardScaler → LR)
    # -------------------------------------------------------------------------
    print("========================================")
    print("TRAINING MODEL 1: LOGISTIC REGRESSION")
    print("========================================")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(random_state=42, max_iter=2000, solver="lbfgs")),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_pipeline, X_val, y_val)

    # Verify predictions
    assert lr_metrics["y_proba"].min() >= 0.0 and lr_metrics["y_proba"].max() <= 1.0
    print("  Logistic Regression trained successfully.")
    print()

    # -------------------------------------------------------------------------
    # 6. MODEL 2 — RANDOM FOREST
    # -------------------------------------------------------------------------
    print("========================================")
    print("TRAINING MODEL 2: RANDOM FOREST")
    print("========================================")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_val, y_val)

    # Verify predictions
    assert rf_metrics["y_proba"].min() >= 0.0 and rf_metrics["y_proba"].max() <= 1.0
    print("  Random Forest trained successfully.")
    print()

    # -------------------------------------------------------------------------
    # 7. CONFUSION MATRICES
    # -------------------------------------------------------------------------
    print("========================================")
    print("CONFUSION MATRICES (Validation Set)")
    print("========================================")
    print()
    print_confusion_matrix("Logistic Regression", lr_metrics)
    print_confusion_matrix("Random Forest", rf_metrics)

    # -------------------------------------------------------------------------
    # 8. MODEL COMPARISON TABLE
    # -------------------------------------------------------------------------
    print("========================================")
    print("MODEL COMPARISON (Validation Set, threshold=0.50)")
    print("========================================")
    print()
    header = f"{'Model':<28} {'Precision':>10} {'Recall':>8} {'F0.5':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    print("-" * len(header))

    def row(name, m):
        return (
            f"{name:<28} {m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{m['f0_5']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>10.4f}"
        )

    print(row("Logistic Regression", lr_metrics))
    print(row("Random Forest", rf_metrics))
    print()

    # -------------------------------------------------------------------------
    # 9. MODEL SELECTION (based on F0.5, then Precision, then Recall)
    # -------------------------------------------------------------------------
    lr_score = (lr_metrics["f0_5"], lr_metrics["precision"], lr_metrics["recall"])
    rf_score = (rf_metrics["f0_5"], rf_metrics["precision"], rf_metrics["recall"])

    if rf_score >= lr_score:
        selected_model_name = "Random Forest"
        selected_model = rf_model
        winner_metrics = rf_metrics
        loser_metrics = lr_metrics
        loser_name = "Logistic Regression"
    else:
        selected_model_name = "Logistic Regression"
        selected_model = lr_pipeline
        winner_metrics = lr_metrics
        loser_metrics = rf_metrics
        loser_name = "Random Forest"

    print("========================================")
    print(f"SELECTED MODEL: {selected_model_name}")
    print("========================================")
    print()
    print(f"REASON:")
    print(f"  {selected_model_name} outperforms {loser_name} on the primary selection criteria:")
    print(f"")
    print(f"  F0.5  : {selected_model_name} = {winner_metrics['f0_5']:.4f}  |  {loser_name} = {loser_metrics['f0_5']:.4f}")
    print(f"  Precision: {selected_model_name} = {winner_metrics['precision']:.4f}  |  {loser_name} = {loser_metrics['precision']:.4f}")
    print(f"  Recall   : {selected_model_name} = {winner_metrics['recall']:.4f}  |  {loser_name} = {loser_metrics['recall']:.4f}")
    print()
    print(f"  Entity resolution is a precision-heavy task: a False Positive (incorrectly")
    print(f"  merging two distinct business records) is typically more harmful than a")
    print(f"  False Negative (missing a true match). F0.5 reflects this trade-off by")
    print(f"  weighting precision twice as heavily as recall.")
    print()

    # -------------------------------------------------------------------------
    # 10. FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    print("========================================")
    print("FEATURE IMPORTANCE")
    print("========================================")
    print()

    # Random Forest importances
    rf_importance = rf_model.feature_importances_

    # Logistic Regression absolute coefficients (from scaler pipeline)
    lr_coef = np.abs(lr_pipeline.named_steps["lr"].coef_[0])

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "random_forest_importance": rf_importance,
        "logistic_regression_absolute_coefficient": lr_coef,
    }).sort_values("random_forest_importance", ascending=False).reset_index(drop=True)

    print("  Random Forest -- Feature Importances (descending):")
    for _, r in importance_df.iterrows():
        bar = "|" * int(r["random_forest_importance"] * 200)
        print(f"    {r['feature']:<40} {r['random_forest_importance']:.4f}  {bar}")
    print()

    print("  Logistic Regression -- Absolute Coefficients (by RF rank):")
    for _, r in importance_df.iterrows():
        bar = "|" * int(r["logistic_regression_absolute_coefficient"] * 3)
        print(f"    {r['feature']:<40} {r['logistic_regression_absolute_coefficient']:.4f}  {bar}")
    print()

    # -------------------------------------------------------------------------
    # 11. SAVE MODELS
    # -------------------------------------------------------------------------
    print("========================================")
    print("SAVING MODELS")
    print("========================================")

    lr_path = MODELS_DIR / "logistic_regression_model.joblib"
    rf_path = MODELS_DIR / "random_forest_model.joblib"
    joblib.dump(lr_pipeline, lr_path)
    joblib.dump(rf_model, rf_path)
    print(f"  Saved: {lr_path}")
    print(f"  Saved: {rf_path}")

    # Mirror to workspace root
    joblib.dump(lr_pipeline, WORKSPACE_ROOT / "models" / "logistic_regression_model.joblib")
    joblib.dump(rf_model, WORKSPACE_ROOT / "models" / "random_forest_model.joblib")
    print()

    # -------------------------------------------------------------------------
    # 12. SAVE RESULTS
    # -------------------------------------------------------------------------
    # model_comparison.tsv
    comparison_df = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "precision": lr_metrics["precision"],
            "recall": lr_metrics["recall"],
            "f0_5": lr_metrics["f0_5"],
            "f1": lr_metrics["f1"],
            "accuracy": lr_metrics["accuracy"],
        },
        {
            "model": "Random Forest",
            "precision": rf_metrics["precision"],
            "recall": rf_metrics["recall"],
            "f0_5": rf_metrics["f0_5"],
            "f1": rf_metrics["f1"],
            "accuracy": rf_metrics["accuracy"],
        },
    ])
    comp_path = RESULTS_DIR / "model_comparison.tsv"
    comparison_df.to_csv(comp_path, sep="\t", index=False)
    print(f"  Saved: {comp_path}")

    # validation_predictions.tsv
    val_pred_df = pd.DataFrame({
        "actual_label": y_val.values,
        "logistic_prediction": lr_metrics["y_pred"],
        "logistic_probability": lr_metrics["y_proba"],
        "random_forest_prediction": rf_metrics["y_pred"],
        "random_forest_probability": rf_metrics["y_proba"],
    })
    val_pred_path = RESULTS_DIR / "validation_predictions.tsv"
    val_pred_df.to_csv(val_pred_path, sep="\t", index=False)
    print(f"  Saved: {val_pred_path}")

    # feature_importance.tsv
    fi_path = MODEL_DATA_DIR / "feature_importance.tsv"
    importance_df.to_csv(fi_path, sep="\t", index=False)
    print(f"  Saved: {fi_path}")

    # Mirror to workspace root
    comparison_df.to_csv(WORKSPACE_ROOT / "model_results" / "model_comparison.tsv", sep="\t", index=False)
    val_pred_df.to_csv(WORKSPACE_ROOT / "model_results" / "validation_predictions.tsv", sep="\t", index=False)
    importance_df.to_csv(WORKSPACE_ROOT / "dataset" / "model_data" / "feature_importance.tsv", sep="\t", index=False)
    print()

    # -------------------------------------------------------------------------
    # 13. SUMMARY
    # -------------------------------------------------------------------------
    print("========================================")
    print("TRAINING COMPLETE — SUMMARY")
    print("========================================")
    print()
    print(f"  Training pairs        : {len(X_train)}")
    print(f"  Validation pairs      : {len(X_val)}")
    print()
    print(f"  Logistic Regression:")
    print(f"    Precision : {lr_metrics['precision']:.4f}")
    print(f"    Recall    : {lr_metrics['recall']:.4f}")
    print(f"    F0.5      : {lr_metrics['f0_5']:.4f}")
    print(f"    F1        : {lr_metrics['f1']:.4f}")
    print(f"    Accuracy  : {lr_metrics['accuracy']:.4f}")
    print()
    print(f"  Random Forest:")
    print(f"    Precision : {rf_metrics['precision']:.4f}")
    print(f"    Recall    : {rf_metrics['recall']:.4f}")
    print(f"    F0.5      : {rf_metrics['f0_5']:.4f}")
    print(f"    F1        : {rf_metrics['f1']:.4f}")
    print(f"    Accuracy  : {rf_metrics['accuracy']:.4f}")
    print()
    print(f"  Selected model        : {selected_model_name}")
    print(f"  Models saved to       : {MODELS_DIR}/")
    print(f"  Results saved to      : {RESULTS_DIR}/")
    print(f"  Feature importance    : {fi_path}")
    print()
    print("  NOTE: Final evaluation on the test partition has NOT been performed.")
    print("        This output represents BASELINE validation-set comparison only.")
    print()


if __name__ == "__main__":
    main()
