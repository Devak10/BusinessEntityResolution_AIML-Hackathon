# Business Entity Resolution — AIML Hackathon

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![RapidFuzz](https://img.shields.io/badge/RapidFuzz-3.0%2B-008080?style=for-the-badge)](https://github.com/maxbachmann/RapidFuzz)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Devak10%2FBusinessEntityResolution--AIML--Hackathon-181717?style=for-the-badge&logo=github)](https://github.com/Devak10/BusinessEntityResolution_AIML-Hackathon)

> **Key Benchmark Performance (Held-out Test Partition):**  
> **Challenge-style Entity-Level Macro $F_{0.5}$:** `0.9802` | **Pair-Level $F_{0.5}$:** `0.9856` | **Pair-Level Precision:** `0.9880` | **Accuracy:** `0.9989`

---

## Executive Summary

**Business Entity Resolution** is the machine learning task of determining whether business records originating from disparate, heterogeneous enterprise channels refer to the same real-world entity. In real-world enterprise databases, identical organizations appear with drastic orthographic differences caused by typographical errors, informal abbreviations, missing street elements, variable corporate legal suffixes (`Pvt Ltd`, `LLC`, `Corp`), and confusing lookalike distractors.

This repository implements an end-to-end, production-grade, leak-free Machine Learning system that:
1. Synthesizes and partitions a **1,000 golden-entity enterprise benchmark** (2,550 records across 3 distinct data feeds) with **zero cluster leakage**.
2. Preprocesses and normalizes raw text using deterministic rule-based algorithms that preserve address numbers and postal codes while standardizing legal entity variations.
3. Implements **candidate blocking** (country clustering and token-similarity filtering) to reduce pairwise comparisons from $O(N^2)$ to high-confidence candidate sets with $>99\%$ recall retention.
4. Computes a robust **15-dimensional similarity feature matrix** combining token-sort ratios, token-set ratios, Jaccard token overlap percentages, Levenshtein distances, and exact match indicators.
5. Trains and validates supervised classifiers (**Logistic Regression** and **Random Forest**), achieving an exceptional **$F_{0.5}$ score of 0.9856** (pair level) and **0.9802** (challenge entity-level macro score) prioritized for high precision.
6. Deploys a full-stack, responsive interactive web dashboard built with **Flask, HTML5, Vanilla CSS, and JavaScript** enabling real-time entity matching, candidate inspection, and live similarity breakdowns.

---

## Key Features

- **End-to-End Leakage-Free Workflow**: Strict cluster-level 80/20 train/test split ensuring no business entity or its variants appear across both training and test partitions.
- **Deterministic Text Normalization**: Rigorous legal entity canonicalization, street suffix expansion/contraction, and digit-preserving spatial token normalization.
- **High-Efficiency Candidate Blocking**: Multi-stage blocking filtering pruning millions of non-viable negative pairs down to high-relevance candidates.
- **15 Engineered Similarity Metrics**: RapidFuzz C++ accelerated Levenshtein, partial ratio, token sort, token set, token overlap (Jaccard), length deltas, and exact match flags.
- **High-Precision ML Decisioning**: Tuned decision thresholds optimizing $F_{0.5}$ metric (weighting precision twice as heavily as recall to eliminate costly false positive mergers in enterprise master data management).
- **Zero-External-API Web Deployment**: Standalone Flask inference dashboard with searchable entity selector, match cards, similarity radars, and confidence scores.

---

## Technologies & Stack

| Layer | Technologies / Libraries | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Core ML engineering, data manipulation, and backend API |
| **Machine Learning** | `scikit-learn` (>=1.3.0) | Logistic Regression, Random Forest, Cross-validation, Metrics |
| **String & NLP Matching** | `rapidfuzz` (>=3.0.0) | High-speed C++ Levenshtein distance, token sort/set ratio, Jaccard overlap |
| **Data Processing** | `pandas` (>=2.0.0), `numpy` (>=1.24.0) | TSV dataset manipulation, feature matrix generation, vectorization |
| **Model Serialization** | `joblib` (>=1.3.0) | Model persistence and fast artifact loading |
| **Web Application** | `flask` (>=3.0.0) | WSGI RESTful backend and template routing |
| **Frontend UI** | HTML5, Vanilla CSS3, Vanilla JavaScript | Responsive dark/light dashboard, real-time live match visualizer |
| **Version Control** | Git & GitHub | Modular codebase and reproducible project artifacts |

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher installed on your system.
- Git installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Devak10/BusinessEntityResolution_AIML-Hackathon.git
cd BusinessEntityResolution_AIML-Hackathon
```

### 2. Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## Execution Guide: Full ML Pipeline

To execute the entire project from dataset generation to model evaluation, run the modular scripts in sequence:

```bash
# 1. Generate master synthetic benchmark datasets (Source 1, 2, 3 and Ground Truth)
python src/create_dataset.py

# 2. Verify schema compliance, identifier integrity, and cross-source linkages
python src/verify_dataset.py

# 3. Perform 80/20 cluster-preserving train/test split with zero leakage
python src/split_dataset.py

# 4. Run data preprocessing and normalization pipelines for train and test partitions
python src/run_preprocessing.py

# 5. Generate blocked candidate pairs and compute 15-dimensional training feature matrix
python src/create_training_data.py

# 6. Train and validate Logistic Regression and Random Forest models
python src/train_models.py

# 7. Evaluate frozen models on unseen test partition and generate challenge metrics
python src/evaluate_test.py
```

### Launch the Interactive Web Application

Start the Flask application server:
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```
- Select any of the 200 held-out Source 1 test entities.
- Inspect real-time candidate generation, 15 feature score vectors, model prediction probabilities, and match resolution cards.

---

## Repository Directory Structure

```
BusinessEntityResolution_AIML-Hackathon/
├── app.py                             # Flask web application server & REST API
├── requirements.txt                   # Production Python package dependencies
├── .gitignore                         # Git exclusion rules (virtualenvs, cache, etc.)
├── README.md                          # Comprehensive technical documentation
│
├── dataset/                           # TSV benchmark datasets
│   ├── source1.tsv                    # Master Golden Catalog (1,000 entities)
│   ├── source2.tsv                    # Master ERP/CRM Feed (800 records)
│   ├── source3.tsv                    # Master Scraped/Web Directory Feed (750 records)
│   ├── ground_truth.tsv               # Master pairwise link table (1,210 relationships)
│   ├── train/                         # 80% Training partition (800 S1 entities)
│   ├── test/                          # 20% Held-out test partition (200 S1 entities)
│   ├── preprocessed/                  # Preprocessed datasets with normalized columns
│   └── model_data/                    # Training candidate pairs and 15 feature vectors
│
├── src/                               # Modular Machine Learning pipeline source code
│   ├── create_dataset.py              # Synthetic enterprise dataset generator (seed=42)
│   ├── verify_dataset.py              # Schema, ID format, and relationship verifier
│   ├── split_dataset.py               # Cluster-preserving train/test splitter
│   ├── load_data.py                   # Reusable TSV partition loaders & data verifiers
│   ├── inspect_data.py                # Exploratory dataset profiling & health checks
│   ├── preprocess.py                  # Text and address normalization functions
│   ├── run_preprocessing.py           # Preprocessing execution pipeline
│   ├── create_training_data.py        # Candidate blocking & feature extraction
│   ├── feature_engineering.py         # 15 similarity feature algorithms
│   ├── train_models.py                # Model training, threshold tuning & comparison
│   └── evaluate_test.py               # Test set evaluation & challenge macro metric scoring
│
├── models/                            # Trained & frozen model artifacts
│   ├── logistic_regression_model.joblib # Production model (F0.5 = 0.9856)
│   └── random_forest_model.joblib       # Champion comparison model
│
├── model_results/                     # Cross-validation & model comparison logs
│   ├── model_comparison.tsv           # Performance comparison across classifiers
│   └── validation_predictions.tsv     # Validation fold predictions
│
├── output/                            # Final test evaluation artifacts
│   ├── candidate_pairs.tsv            # Test candidates with prediction probabilities
│   ├── matching_results.tsv           # Official 200-entity submission predictions
│   ├── test_candidate_features.tsv    # 15 feature vectors for test candidates
│   ├── entity_level_scores.tsv        # Per-entity macro precision, recall, F0.5
│   └── final_test_metrics.tsv         # Final summary metrics table
│
├── static/                            # Frontend assets
│   ├── css/style.css                  # Modern UI styles, dark mode, responsive layout
│   └── js/app.js                      # Client-side dynamic interaction & API integration
│
└── templates/                         # HTML templates
    └── index.html                     # Interactive entity resolution dashboard
```

---

## 1. Why the Dataset Was Scaled to 1,000 Source 1 Entities

## 3. Dataset Specifications & Schemas

All files are strictly **Tab-Separated Values (`.tsv`)** using `\t` as the delimiter (no comma-delimited CSVs).

### Source Datasets
Every source dataset shares a uniform 4-column schema:

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `entity_id` | String | Unique source-prefixed identifier | `S1-00001`, `S2-00034`, `S3-00315` |
| `business_name` | String | Name of the business organization | `Apex Logistics Private Limited` |
| `business_address` | String | Commercial or registered address | `Plot 18, Road No. 9, MIDC Industrial Area, Mumbai` |
| `country` | String | Country of operation (`India` or `USA`) | `India` or `USA` |

### Preprocessed Datasets (7 Columns)
In addition to the 4 original columns (which remain strictly preserved), preprocessed datasets append 3 normalized feature columns:

| Column Name | Type | Description | Normalized Output Example |
| :--- | :--- | :--- | :--- |
| `business_name_normalized` | String | Normalized business name | `apex logistics pvt ltd` |
| `business_address_normalized` | String | Normalized address with preserved digits | `plot 18 rd no 9 midc indl area andheri east mumbai maharashtra 400093` |
| `country_normalized` | String | Lowercase standardized country | `india` |

### Ground Truth
The ground truth files define the true pairwise identity linkages:

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `source1_entity_id` | String | ID of the Source 1 business | `S1-00001` |
| `matched_entity_ids` | String | Comma-separated matching IDs from S2 and/or S3, or empty if singleton | `S2-00034,S3-00315` or `S2-00105` or `""` |

---

## 4. Master Dataset Summary Statistics

```
========================================
BUSINESS ENTITY RESOLUTION DATASET
==================================

Source 1 records: 1000
Source 2 records: 800
Source 3 records: 750

Total records: 2550

Matching Source 1 entities: 700 (70.0%) [Target: 65-75%]
Singleton Source 1 entities: 300 (30.0%) [Target: 25-35%]

Matches in BOTH: 380 (38.0%)
Matches in Source 2 ONLY: 180 (18.0%)
Matches in Source 3 ONLY: 140 (14.0%)

Total matching relationships: 1210

Country distribution:
Source 1: {'India': 500, 'USA': 500}
Source 2: {'India': 435, 'USA': 365}
Source 3: {'India': 405, 'USA': 345}
========================================
```

---

## 5. Train / Test Split Strategy

Entity resolution datasets cannot be split by independently partitioning individual tables. Doing so would place one record of a company in training and another record of the same company in testing, causing severe data leakage and artificially inflated metrics.

### Key Principles of the Split
1. **Source 1 Entity-Level Partitioning**:
   * Splitting occurs strictly at the Source 1 business level with random seed `42`.
   * **80% (800 entities)** are assigned to `train`.
   * **20% (200 entities)** are assigned to `test`.
2. **Entity Cluster Preservation**:
   * For every Source 1 entity assigned to `train`, all corresponding Source 2 and Source 3 records linked through `ground_truth.tsv` are placed into the `train` partition (`source2_train.tsv`, `source3_train.tsv`).
   * For every Source 1 entity assigned to `test`, all corresponding Source 2 and Source 3 records linked through `ground_truth.tsv` are placed into the `test` partition (`source2_test.tsv`, `source3_test.tsv`).
   * No entity cluster crosses between train and test.
3. **Prevention of Information Leakage**:
   * An explicit automated data leakage test verifies that:
     $$\text{Linked}(S2_{\text{train}}) \cap \text{Linked}(S2_{\text{test}}) = \emptyset$$
     $$\text{Linked}(S3_{\text{train}}) \cap \text{Linked}(S3_{\text{test}}) = \emptyset$$
   * A model evaluated on the test set is tested on completely unseen real-world businesses.
4. **Deterministic Distractor Handling**:
   * Unlinked distractor records in Source 2 and Source 3 are deterministically split 80% train / 20% test using seed `42` so that both sets reflect genuine noise levels.
5. **Offline Evaluation Role of Test Ground Truth**:
   * `ground_truth_test.tsv` is strictly reserved for offline benchmarking and evaluation.
   * During prediction / inference, the ML model receives only `source1_test.tsv`, `source2_test.tsv`, and `source3_test.tsv` and must discover the linkages autonomously.

### Train / Test Split Summary Table

| Metric | Master Dataset | Train Partition (80%) | Test Partition (20%) |
| :--- | :--- | :--- | :--- |
| **Source 1 Records** | 1,000 | **800** (80.0%) | **200** (20.0%) |
| **Source 2 Records** | 800 | **633** (79.1%) | **167** (20.9%) |
| **Source 3 Records** | 750 | **597** (79.6%) | **153** (20.4%) |
| **Total Matched Relationships** | 1,210 | **958** | **252** |
| **Singleton Source 1 Entities** | 300 | **247** | **53** |
| **Matching Source 1 Entities** | 700 | **553** | **147** |

---

## 6. Dataset Loading and Inspection (Step 3)

### Overview
Step 3 implements a standardized data loading module ([`load_data.py`](file:///d:/BusinessEntityResolution_AIML-Hackathon/business-entity-resolution/src/load_data.py)) and comprehensive inspection suite ([`inspect_data.py`](file:///d:/BusinessEntityResolution_AIML-Hackathon/business-entity-resolution/src/inspect_data.py)).

### Key Architectural Guidelines
* **Strict TSV Parsing**: All files are explicitly parsed with `pandas.read_csv(filepath, sep="\t")`. Comma-separated CSV loading is strictly forbidden.
* **Partition Separation**: Train and test datasets are loaded via distinct functions (`load_train_data()` and `load_test_data()`).
* **Automated Schema Verification**: Every file load verifies the presence of required columns and raises clear descriptive errors if mismatched.
* **Exploratory Profiling**:
  * Displays shapes, data types, missing value counts, and unique value counts.
  * Calculates dynamic country distributions via `value_counts()` without hardcoding country categories.
  * Verifies data quality: zero missing values in mandatory fields, zero duplicate entity IDs, valid prefixes (`S1-`, `S2-`, `S3-`), and singleton counting.
* **Train / Test Consistency**: Confirms zero overlap across entity IDs between partitions and identical schemas.

---

## 7. Data Preprocessing (Step 4)

### Why Entity Resolution Requires Normalization
In disparate corporate data sources, the exact same real-world business entity rarely appears with identical string representations. Variations arise from:
- **Capitalization**: Mainframes export in `ALL CAPS`, web scrapes in `lowercase`, and ERPs in `Title Case`.
- **Legal Form Variations**: `Private Limited`, `Pvt Ltd`, `Pvt. Ltd.`, `P. Ltd.`, `Ltd`, `Limited`, `LLC`, `L.L.C.`, `Corporation`, `Corp.`, `Inc.`.
- **Punctuation & Whitespace**: Irregular hyphens, slashes, stray spaces, and periods.
- **Address Abbreviations**: `Road` vs `Rd`, `Street` vs `St`, `Avenue` vs `Ave`, `Suite` vs `Ste` vs `#`.
Without normalization, string comparison algorithms (Levenshtein, Jaro-Winkler, TF-IDF n-grams) suffer significant recall degradation due to trivial formatting mismatches.

### Preprocessing Architecture & Rules
1. **Original Columns Preserved**:
   The raw columns (`entity_id`, `business_name`, `business_address`, `country`) are never overwritten.
2. **Normalized Columns Appended**:
   Three clean feature columns are appended:
   - `business_name_normalized`
   - `business_address_normalized`
   - `country_normalized`
3. **Synthetic Generator Marker Removal**:
   Artificial generator markers (e.g. `distractor`, `scraped`) introduced during synthetic generation are stripped case-insensitively from `business_name_normalized` so they never serve as artificial ML shortcuts.
4. **Digit Preservation**:
   All digits in addresses (street numbers, suites, floors, PIN/ZIP codes) are strictly preserved (e.g., `2100 North Stemmons Freeway, Suite 450` preserves `2100` and `450`).
5. **Dynamic Country Handling**:
   Standardizes aliases (e.g., `United States` → `usa`), while dynamically lowercasing and trimming any foreign country string without hardcoded exclusions.
6. **Zero Data Leakage**:
   No ground truth labels (`ground_truth_train.tsv`, `ground_truth_test.tsv`) or target match outcomes are accessed during preprocessing.
7. **Consistent Pipeline Across Train & Test**:
   The identical `preprocess_source_dataframe()` function processes both train and test partitions without modifications.
8. **No External Databases**:
   No geocoding APIs, external internet lookups, or third-party business directories are used.

### Preprocessing Before / After Examples

| Source & ID | Original Field | Normalized Field |
| :--- | :--- | :--- |
| **S2-00001** (Name) | `Bright Software Systems Enterprises Ventures Inc. Distractor` | `bright software systems enterprises ventures inc` |
| **S2-00001** (Address) | `Ground Fl. , Salt Lake Sector V , Sahar Airport Rd, Mumbai, Maharashtra 400017` | `ground fl salt lake sec v sahar airport rd mumbai maharashtra 400017` |
| **S2-00002** (Name) | `VERITAS TECH PLATFORMS GROUP INC` | `veritas tech platforms group inc` |
| **S2-00002** (Address) | `189 FINANCIAL CENTER PKWY, UNIT 1329, CHICAGO, IL 60652` | `189 financial center pkwy unit 1329 chicago il 60652` |
| **S3-00005** (Name) | `global bright software systems enterprises p. ltd. scraped` | `global bright software systems enterprises pvt ltd` |
| **S3-00005** (Address) | `ground fl., salt lake sector v, sahar airport rd., mumbai, behind city center mall, maharashtra 400017` | `ground fl salt lake sec v sahar airport rd mumbai behind city center mall maharashtra 400017` |
| **S1-00001** (Name) | `Apex Logistics Private Limited` | `apex logistics pvt ltd` |
| **S1-00001** (Address) | `Plot 18, Road No. 9, MIDC Industrial Area, Andheri East, Mumbai, Maharashtra 400093` | `plot 18 rd no 9 midc indl area andheri east mumbai maharashtra 400093` |

---

## 8. Feature Engineering and Training Pair Creation

### Why Entity Pairs Are Required for Supervised Entity Resolution
Entity resolution cannot be framed directly as a standard single-record multiclass classification problem because entity clusters vary widely in size, new unseen businesses arrive continuously, and cross-source record schemas lack shared global keys. By converting the resolution task into pairwise classification ($s_1 \leftrightarrow s_j$), the model learns a generic similarity function: predicting whether any given pair of records refers to the same real-world entity (`label = 1`) or different entities (`label = 0`).

### How Positive Pairs Are Obtained from Ground Truth
Positive training pairs (`label = 1`) are derived strictly from `ground_truth_train.tsv` without accessing test ground truth. For every verified match between Source 1 and Source 2 (497 pairs) and between Source 1 and Source 3 (461 pairs), a positive pair is formed:
`[source1_entity_id, source_entity_id, source_type, label=1]`. Total positive pairs: `958`. Singletons in Source 1 naturally produce no positive pairs.

### How Negative Candidates Are Generated
Negative pairs (`label = 0`) cannot be created through naive random pairing because millions of random non-matches are trivial for an ML classifier (e.g., completely disparate names in different countries) and create severe class imbalance. We implement candidate blocking:
- **Primary blocking**: Partition candidates by `country_normalized`.
- **Candidate scoring**: Within the country block, measure token-sort and token-set similarities on names and addresses.
- **Filtering**: Exclude all known positive matches (`known_positives`).
- **Stratified selection**: Separate candidates into hard negatives and easy negatives, sampling across all 800 Source 1 training entities to ensure comprehensive coverage.

### Why Hard Negatives Are Important
Hard negatives force the ML model to learn subtle boundary discriminations rather than relying on shallow heuristics. A hard negative shares strong surface resemblance—such as co-located office addresses in the same tech park, sister corporate entities under the same brand umbrella, or synthetic distractors—yet represents distinct legal organizations. Without hard negatives, the classifier would suffer high false-positive rates when deployed against real-world dirty feeds.

### Why Candidate Blocking Is Used
Pairwise all-vs-all matching scales quadratically ($\mathcal{O}(N \times M)$), generating over 1,000,000 pairs for train data alone. Candidate blocking prunes the vast space of obvious non-matches, focusing computational effort on plausible candidate pairs.

### Why Entity IDs and Ground-Truth Labels Are Not ML Features
Entity IDs (e.g. `S1-00001`, `S2-00034`) and ground truth linkage strings (`matched_entity_ids`) are identifiers or labels, not real-world attributes. Including them in feature matrix $X$ would cause severe data leakage and overfitting: the model would memorize ID strings rather than learning transferable patterns of name typographical variation, word transposition, abbreviation equivalence, and address matching. In `training_features.tsv`, only the 15 numeric similarity features and the target `label` column exist.

### The 15 Engineered Features (Consistent 0–100 Scale for Similarity Scores)
All similarity metrics are computed on a consistent 0.0 to 100.0 scale:
1. `name_ratio`: Levenshtein character edit-distance ratio (0.0 to 100.0).
2. `name_token_sort_ratio`: Token-sorted edit distance ratio, invariant to word reordering (0.0 to 100.0).
3. `name_token_set_ratio`: Token-set edit distance ratio, robust to subset/superset naming (0.0 to 100.0).
4. `name_partial_ratio`: Substring similarity ratio for partial brand titles (0.0 to 100.0).
5. `address_ratio`: Character edit distance ratio between normalized addresses (0.0 to 100.0).
6. `address_token_sort_ratio`: Token-sorted edit distance ratio for reordered address tokens (0.0 to 100.0).
7. `address_token_set_ratio`: Token-set edit distance ratio for landmark-supplemented addresses (0.0 to 100.0).
8. `address_partial_ratio`: Substring ratio for shared physical street lines (0.0 to 100.0).
9. `country_match`: Binary indicator whether normalized country strings match (1 or 0).
10. `name_length_difference`: Absolute character length difference between names (Integer $\ge 0$).
11. `address_length_difference`: Absolute character length difference between addresses (Integer $\ge 0$).
12. `name_token_overlap`: Jaccard token overlap percentage $(|A \cap B| / |A \cup B|) \times 100.0$ (0.0 to 100.0).
13. `address_token_overlap`: Jaccard token overlap percentage $(|A \cap B| / |A \cup B|) \times 100.0$ (0.0 to 100.0).
14. `name_exact_match`: Binary indicator whether normalized names are character-identical (1 or 0).
15. `address_exact_match`: Binary indicator whether normalized addresses are character-identical (1 or 0).

---

## 9. Hard Negative Design

This dataset embeds **60 explicit Hard Negative pairs (120 businesses)** in Source 1, plus **over 100 hard negative distractors** across Source 2 and Source 3:

1. **Cross-Country Brand Collisions**: (e.g., `Apex Logistics Private Limited`, Mumbai vs `Apex Logistics LLC`, Dallas).
2. **Intra-City Similar Name Collisions**: (e.g., `Vertex Solutions India Pvt Ltd` vs `Vertex Financial Solutions Pvt Ltd`, Bengaluru).
3. **Co-Located Tech Park / Building Collisions**: (e.g., `Quantum Labs Cyber Security` Unit 402 vs `NextGen FinTech Solutions` Unit 403, DLF Cyber City).
4. **Conglomerate Sister Companies**: (e.g., `Tata Consumer Products Ltd` vs `Tata Motors Commercial Vehicles Ltd`, Mumbai).

---

## 10. Why Synthetic Data Is Used

1. **Controlled Ground Truth**: True identity linkages are 100% verified.
2. **Customized Stress-Testing**: Systematic error injection (typos, omitted PINs, reordered addresses, hard negatives).
3. **Data Privacy & Compliance**: Zero GDPR / DPDP exposure.
4. **Deterministic Reproducibility**: Seeded generation (`seed=42`) ensures identical benchmarks across environments.

---

## 11. How to Regenerate, Inspect, Preprocess, Build Features & Verify

### Requirements
* Python 3.10+
* `pandas`
* `numpy`
* `RapidFuzz`

Install dependencies:
```bash
pip install -r requirements.txt
```

### Step 1: Generate Master Dataset
```bash
python business-entity-resolution/src/create_dataset.py
```

### Step 2: Perform Train / Test Split
```bash
python business-entity-resolution/src/split_dataset.py
```

### Step 3: Inspect Raw Partitions
```bash
python business-entity-resolution/src/inspect_data.py
```

### Step 4: Run Data Preprocessing
```bash
python business-entity-resolution/src/run_preprocessing.py
```

### Step 5: Generate Training Pairs & Features
```bash
python business-entity-resolution/src/create_training_data.py
```

### Step 6: Train ML Models
```bash
.venv\Scripts\python business-entity-resolution/src/train_models.py
```

### Master Verification Suite
```bash
python business-entity-resolution/src/verify_dataset.py
```

---

## 12. Machine Learning Models

### Why Two Algorithms Are Trained
This project trains and compares two fundamentally different supervised classifiers on the same 15 engineered similarity features, allowing a rigorous side-by-side evaluation. Both models solve an identical binary classification task: given a pair of business records, predict `label = 1` (same real-world entity) or `label = 0` (different entities).

Both models are trained exclusively on `training_features.tsv` using an internal 80/20 stratified validation split (`random_state=42`). **No test partition data is accessed at any point during training or model selection.**

---

### Model 1: Logistic Regression

**Why Logistic Regression:**
- **Simple interpretable baseline**: Coefficients directly reveal which features drive predictions.
- **Fast training and inference**: Ideal for lightweight student demonstrations.
- **Provides calibrated probability estimates**: Enables threshold tuning in later steps.
- **Effective on approximately linearly separable features**: RapidFuzz similarity scores on a 0-100 scale are well-suited after `StandardScaler` normalization.

**Implementation**: A scikit-learn `Pipeline` wrapping `StandardScaler` → `LogisticRegression(max_iter=2000, random_state=42)` is saved as a single `.joblib` file, preserving scaler state for inference.

---

### Model 2: Random Forest

**Why Random Forest:**
- **Handles nonlinear relationships**: Captures complex interactions between name and address similarity scores.
- **Robust to feature scale differences**: No standardization required (`class_weight="balanced"` handles imbalance).
- **Captures feature interactions**: E.g. moderate `name_token_set_ratio` combined with high `address_token_overlap` may jointly confirm a match.
- **Provides interpretable feature importances**: Ranks all 15 similarity signals by predictive contribution.

**Implementation**: `RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)` trained directly on raw features.

---

### Validation Results (Threshold = 0.50)

Both models evaluated on the same 20% held-out validation split (672 pairs: 192 positive, 480 negative). Primary selection metric: **F0.5** (weights precision twice as heavily as recall), then **Precision**, then **Recall**.

| Model | Precision | Recall | F0.5 | F1 | Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **Logistic Regression** | **0.9495** | 0.9792 | **0.9553** | 0.9641 | 0.9792 |
| Random Forest | 0.9453 | **0.9896** | 0.9538 | **0.9669** | **0.9807** |

**Selected Model: Logistic Regression**

Logistic Regression achieves a higher F0.5 (0.9553 vs 0.9538) and higher Precision (0.9495 vs 0.9453). In entity resolution, **False Positives** (incorrectly merging distinct records) are more damaging than False Negatives (missing a match), making F0.5 and Precision the decisive criteria.

---

### Feature Importance Rankings

| Rank | Feature | RF Importance | LR Abs. Coefficient |
| :---: | :--- | ---: | ---: |
| 1 | `name_ratio` | 0.1959 | 4.7830 |
| 2 | `name_token_sort_ratio` | 0.1622 | 3.3304 |
| 3 | `address_token_overlap` | 0.1418 | 0.5767 |
| 4 | `address_token_set_ratio` | 0.1371 | 2.4743 |
| 5 | `name_token_set_ratio` | 0.0878 | 1.6352 |
| 6 | `address_token_sort_ratio` | 0.0734 | 0.5479 |
| 7 | `address_ratio` | 0.0534 | 0.6288 |
| 8 | `name_length_difference` | 0.0451 | 1.2735 |
| 9 | `name_partial_ratio` | 0.0337 | 4.1273 |
| 10 | `name_token_overlap` | 0.0304 | 1.1294 |
| 11 | `address_partial_ratio` | 0.0268 | 0.2038 |
| 12 | `name_exact_match` | 0.0078 | 0.2461 |
| 13 | `address_length_difference` | 0.0038 | 0.1987 |
| 14 | `address_exact_match` | 0.0009 | 0.1576 |
| 15 | `country_match` | 0.0000 | 0.0000 |

> **Note**: `country_match` shows zero importance within training because candidate blocking grouped all pairs by the same country. It remains in the feature set for cross-country generalization at inference time.

---

### Saved Model Artifacts

| File | Description |
| :--- | :--- |
| `models/logistic_regression_model.joblib` | Complete pipeline (StandardScaler + LogisticRegression) |
| `models/random_forest_model.joblib` | Random Forest classifier |
| `model_results/model_comparison.tsv` | Validation metrics for both models |
| `model_results/validation_predictions.tsv` | Per-pair predictions and probabilities on the validation set |
| `dataset/model_data/feature_importance.tsv` | Feature importance comparison table |

---

## 13. Final Test Evaluation and Output

### Methodology
The saved Logistic Regression model was **frozen** after validation-set selection. No retraining, parameter tuning, or threshold adjustment was performed on test data. The process strictly follows:

1. **Load** the trained `logistic_regression_model.joblib` (no retraining).
2. **Load** the preprocessed test partition (`source1/2/3_test_preprocessed.tsv`).
3. **Generate candidate pairs** using the identical country-based blocking strategy used during training.
4. **Compute the same 15 features** for all candidates.
5. **Predict** using threshold = 0.50. Ground truth is NOT consulted at this step.
6. **Load** `ground_truth_test.tsv` ONLY AFTER predictions are complete — for offline evaluation only.
7. **Write** final output files.

### Output Files

| File | Description |
| :--- | :--- |
| `output/candidate_pairs.tsv` | All candidates considered, with prediction probability and label |
| `output/matching_results.tsv` | One row per Source 1 test entity; comma-separated predicted matches |
| `output/test_candidate_features.tsv` | Full feature matrix for test candidates |
| `output/final_test_metrics.tsv` | Pair-level and entity-level metrics |
| `output/entity_level_scores.tsv` | Per-entity precision, recall, F0.5 |

### Output Format Rules
- `matching_results.tsv` contains **exactly 200 rows** — one per Source 1 test entity.
- Multiple predicted matches are **comma-separated** in `matched_entity_ids`.
- Singleton entities (no predicted matches) have an **empty** `matched_entity_ids` field.
- All predicted IDs begin with `S2-` or `S3-` and exist in the test partition.

### Final Test Results (Threshold = 0.50)

#### Pair-Level Metrics

| Metric | Value |
| :--- | ---: |
| Precision | **0.9880** |
| Recall | 0.9762 |
| **F0.5** | **0.9856** |
| F1 | 0.9820 |
| Accuracy | 0.9989 |
| TP | 246 |
| TN | 8,311 |
| FP | 3 |
| FN | 6 |

#### Entity-Level Metrics (Challenge-Style)

| Metric | Value |
| :--- | ---: |
| Source 1 test entities | 200 |
| Entities with true matches | 147 |
| True singletons | 53 |
| Entities with predicted matches | 144 |
| Predicted singletons | 56 |
| Macro Precision | 0.9962 |
| Macro Recall | 0.9800 |
| **Challenge-style Macro F0.5** | **0.9802** |

> The entity-level Macro F0.5 is the **primary challenge evaluation metric**. It is computed per Source 1 entity and averaged across all 200 test entities. True-singleton entities that are also predicted as singletons contribute a score of 1.0.

### Comparison: Validation vs. Test

| Stage | F0.5 |
| :--- | ---: |
| Validation (Logistic Regression) | 0.9553 |
| Final Test Pair-level | 0.9856 |
| Final Test Entity-level Macro | **0.9802** |

Values are reported as measured with no post-hoc adjustment. The improvement on the test set reflects the quality and generalization of the trained features — no test data was used for model selection or tuning.

## 14. Web Deployment

A lightweight, modern web application built with **Flask**, **HTML5**, **Vanilla CSS**, and **Vanilla JavaScript** provides an interactive interface for evaluating cross-source entity matches.

### Key Deployment Characteristics

- **Framework**: Flask (Python WSGI micro-framework) with zero heavy client-side frameworks (no React, Angular, Vue, or Bootstrap).
- **Inference Pipeline**:
  1. The user selects a Source 1 business entity from a searchable selector of all 200 held-out test entities.
  2. The system filters candidate records from Source 2 and Source 3 by matching normalized country.
  3. Candidate blocking filters are applied using identical similarity heuristics from Step 7 (`token_sort_ratio >= 50`, `token_set_ratio >= 65`, or `address_sort_ratio >= 50`).
  4. The exact same 15 numerical similarity features are calculated in identical order.
  5. The frozen **Logistic Regression** model (`models/logistic_regression_model.joblib`) computes real-time class 1 probabilities via `model.predict_proba(X)[:, 1]`.
  6. Matches with probability $\ge 0.50$ are displayed as matching cards with status badges and exact percentage probabilities.
- **Data Integrity & Isolation**:
  - The application works entirely with the existing local preprocessed synthetic dataset and frozen model.
  - **No external APIs, databases, web scraping, geocoding, or government registries** are utilized.
  - **No ground truth files** are loaded or accessed anywhere in `app.py`, ensuring complete zero-leakage inference.

### Running the Web Application

To launch the web application locally:

```bash
python app.py
```

Then open your browser and navigate to:

```text
http://127.0.0.1:5000
```

