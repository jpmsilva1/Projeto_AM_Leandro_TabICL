# Kaggle Pipeline for TabArena Evaluation — Group 2: TabICL v2

> [!IMPORTANT]
> **This is v2 of the pipeline.** It fixes 12+ critical bugs found in v1 during a thorough audit. Do NOT use the previous version.

## Kaggle Setup

1. Create a new Notebook on Kaggle.
2. Go to **Session options** → **Accelerator** → select **GPU T4 x2** (or P100).
3. Toggle **Internet** to **ON**.
4. Set **Persistence** to **Files only** (so your CSV results survive if the session restarts).

---

## Cell 1 — Install Dependencies

```python
# Install required packages. The red dependency warnings are normal on Kaggle — ignore them.
!pip install -q tabicl "pytabkit[models]" openml optuna scikit-learn lightgbm xgboost catboost
# AutoGluon is large; install separately
!pip install -q "autogluon.tabular[all]"

# Clear corrupted OpenML cache from previous failed downloads
!rm -rf ~/.cache/openml
!rm -rf /root/.cache/openml
```

---

## Cell 2 — Full Pipeline

Copy-paste the entire block below into a single Kaggle cell.

```python
# ==============================================================================
# TabArena Evaluation Pipeline — Group 2: TabICL v2
# ==============================================================================
# This script evaluates TabICL v2 against LightGBM_TD, XGBoost_TD, CatBoost_TD,
# and AutoGluon (best_quality) across the approved TabArena classification datasets.
#
# Results are saved incrementally to CSV after EACH model×dataset evaluation,
# so even if the session crashes, partial results are preserved.
# ==============================================================================

import os
import gc
import csv
import time
import shutil
import warnings
import tempfile

import numpy as np
import pandas as pd
import openml

# Force OpenML to use a clean cache directory to bypass corrupted root cache
openml.config.cache_directory = os.path.expanduser('/kaggle/working/openml_cache')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ---------- reproducibility ----------
SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# ---------- GPU detection ----------
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ==============================================================================
# 1. DATASET CONFIGURATION
# ==============================================================================
# These are the approved classification datasets from the TabArena curation CSV
# (Final Decision = "Yes", task_type = "Supervised Classification", complete metadata).
#
# We have 16 classification datasets with complete task IDs:
#   - 3 small  (n < 1,000)
#   - 6 medium (1,000 ≤ n ≤ 10,000)
#   - 7 large  (n > 10,000)

DATASETS = [
    # --- SMALL (n < 1,000) ---
    {"tid": 359955, "name": "blood-transfusion",   "n": 748,    "classes": 2,  "features": 5,    "regime": "small"},
    {"tid": 37,     "name": "diabetes",             "n": 768,    "classes": 2,  "features": 9,    "regime": "small"},
    {"tid": 2,      "name": "anneal",               "n": 898,    "classes": 5,  "features": 39,   "regime": "small"},
    # --- MEDIUM (1,000 ≤ n ≤ 10,000) ---
    {"tid": 168757, "name": "credit-g",             "n": 1000,   "classes": 2,  "features": 21,   "regime": "medium"},
    {"tid": 359956, "name": "qsar-biodeg",          "n": 1055,   "classes": 2,  "features": 42,   "regime": "medium"},
    {"tid": 45,     "name": "splice",               "n": 3190,   "classes": 3,  "features": 61,   "regime": "medium"},
    {"tid": 359967, "name": "Bioresponse",          "n": 3751,   "classes": 2,  "features": 1777, "regime": "medium"},
    {"tid": 3892,   "name": "hiva_agnostic",        "n": 4229,   "classes": 2,  "features": 1618, "regime": "medium"},
    {"tid": 359968, "name": "churn",                "n": 5000,   "classes": 2,  "features": 21,   "regime": "medium"},
    # --- LARGE (n > 10,000) ---
    {"tid": 3688,   "name": "houses",               "n": 20640,  "classes": 2,  "features": 9,    "regime": "large"},
    {"tid": 359979, "name": "Amazon_employee",      "n": 32769,  "classes": 2,  "features": 10,   "regime": "large"},
    {"tid": 3945,   "name": "KDDCup09_appetency",   "n": 50000,  "classes": 2,  "features": 231,  "regime": "large"},
    {"tid": 168868, "name": "APSFailure",           "n": 76000,  "classes": 2,  "features": 171,  "regime": "large"},
    {"tid": 361329, "name": "KDD98",                "n": 82318,  "classes": 2,  "features": 478,  "regime": "large"},
    {"tid": 211986, "name": "Diabetes130US",        "n": 101766, "classes": 3,  "features": 50,   "regime": "large"},
    {"tid": 360113, "name": "porto-seguro",         "n": 595212, "classes": 2,  "features": 58,   "regime": "large"},
]

# ----- Batching control -----
# If Kaggle times out, change these to process a subset.
# Example: BATCH_START=0, BATCH_END=5 processes the first 5 datasets.
BATCH_START = 0
BATCH_END = len(DATASETS)  # process all
datasets_to_run = DATASETS[BATCH_START:BATCH_END]

RESULTS_FILE = "/kaggle/working/kaggle_results.csv"
METADATA_FILE = "/kaggle/working/dataset_metadata.csv"

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def safe_cleanup():
    """Aggressive memory cleanup for Kaggle GPU sessions."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(0.5)


def load_openml_dataset(task_id):
    """Load a dataset from OpenML by task ID."""
    task = openml.tasks.get_task(task_id, download_data=True)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    return X, y, categorical_indicator, attribute_names, dataset.name


def preprocess(X, y, categorical_indicator):
    """
    Preprocess features and target for sklearn-compatible models.
    """
    X = X.copy()
    
    # Encode categorical columns
    for i, col in enumerate(X.columns):
        if categorical_indicator[i] or X[col].dtype == "object" or str(X[col].dtype) == "category":
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    
    # Convert all to float
    X = X.astype(float)
    
    # 1. Drop completely empty columns (prevents median() from returning NaN)
    X = X.dropna(axis=1, how='all')
    
    # 2. Fill missing values (median first, then 0 for any edge cases)
    X = X.fillna(X.median()).fillna(0)
    
    # 3. Drop constant columns (prevents internal boolean indexing bugs in TabICL)
    X = X.loc[:, (X != X.iloc[0]).any()]
    
    # Encode target labels as integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(np.asarray(y).ravel())
    
    return X, y_encoded, le


def g_mean_score(y_true, y_pred):
    """G-Mean: geometric mean of per-class recall (same formula as the project template)."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if not mask.any():
            continue
        recalls.append(float((y_pred[mask] == c).mean()))
    if not recalls:
        return 0.0
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0)))))


def compute_auc(y_true, y_proba, n_classes):
    """
    Compute AUC correctly for binary and multiclass cases.
    - Binary: use y_proba[:, 1] (probability of positive class).
    - Multiclass: use y_proba with multi_class='ovo'.
    """
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovo"))
    except Exception as e:
        print(f"      ⚠️ AUC failed: {e}")
        return np.nan


def compute_cross_entropy(y_true, y_proba, classes):
    """Compute cross-entropy with explicit label ordering."""
    try:
        return float(log_loss(y_true, y_proba, labels=classes))
    except Exception as e:
        print(f"      ⚠️ CE failed: {e}")
        return np.nan


# ==============================================================================
# 3. MODEL BUILDERS
# ==============================================================================

def build_tabicl():
    """TabICL v2 — the group's assigned model."""
    from tabicl import TabICLClassifier
    # TabICL doesn't take seed or random_state in __init__
    return TabICLClassifier(device=DEVICE)


def build_lightgbm_td():
    """LightGBM with meta-tuned defaults from pytabkit."""
    from pytabkit import LGBM_TD_Classifier
    return LGBM_TD_Classifier(random_state=SEED)


def build_xgboost_td():
    """XGBoost with meta-tuned defaults from pytabkit."""
    from pytabkit import XGB_TD_Classifier
    return XGB_TD_Classifier(random_state=SEED)


def build_catboost_td():
    """CatBoost with meta-tuned defaults from pytabkit."""
    from pytabkit import CatBoost_TD_Classifier
    return CatBoost_TD_Classifier(random_state=SEED)


# Model registry: name -> (builder_function, needs_special_handling)
MODELS = {
    "TabICL":      (build_tabicl,      False),
    "LightGBM_TD": (build_lightgbm_td, False),
    "XGBoost_TD":  (build_xgboost_td,  False),
    "CatBoost_TD": (build_catboost_td, False),
    "AutoGluon":   (None,              True),  # handled separately
}


# ==============================================================================
# 4. EVALUATION FUNCTIONS
# ==============================================================================

def evaluate_sklearn_model(model_name, model, X_train, X_test, y_train, y_test, n_classes):
    """Evaluate an sklearn-compatible model (TabICL, LightGBM_TD, XGBoost_TD, CatBoost_TD)."""
    print(f"    🔧 {model_name}...", end=" ", flush=True)
    
    # --- FIT ---
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    
    # --- PREDICT ---
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    predict_time = time.perf_counter() - t0
    
    total_time = fit_time + predict_time
    
    # --- METRICS ---
    acc = accuracy_score(y_test, y_pred)
    gmean = g_mean_score(y_test, y_pred)
    
    classes = np.arange(n_classes)
    auc = compute_auc(y_test, y_proba, n_classes) if y_proba is not None else np.nan
    ce = compute_cross_entropy(y_test, y_proba, classes) if y_proba is not None else np.nan
    
    print(f"ACC={acc:.4f} | AUC={auc:.4f} | Time={total_time:.1f}s")
    
    return {
        "ACC": acc, "AUC_OVO": auc, "G_Mean": gmean, "CE": ce,
        "fit_time_s": fit_time, "predict_time_s": predict_time, "total_time_s": total_time,
    }


def evaluate_autogluon(X_train, X_test, y_train, y_test, n_classes, dataset_name):
    """Evaluate AutoGluon TabularPredictor with best_quality preset."""
    from autogluon.tabular import TabularPredictor
    
    print(f"    🔧 AutoGluon...", end=" ", flush=True)
    
    # Create a unique temporary directory for this run
    ag_path = tempfile.mkdtemp(prefix=f"ag_{dataset_name}_")
    
    # Prepare data: AutoGluon needs a DataFrame with the label column
    train_df = X_train.copy()
    train_df["__target__"] = y_train
    
    test_df = X_test.copy()
    # Do NOT add the target to test data
    
    t0 = time.perf_counter()
    
    ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
    predictor = TabularPredictor(
        label="__target__",
        eval_metric=ag_metric,
        path=ag_path,
        verbosity=0,
    )
    predictor.fit(
        train_data=train_df,
        presets="best_quality",
        time_limit=1800,  # 30 min max per dataset to stay within Kaggle limits
    )
    fit_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_pred = predictor.predict(test_df).values
    y_proba_df = predictor.predict_proba(test_df)
    predict_time = time.perf_counter() - t0
    
    total_time = fit_time + predict_time
    
    # Convert AutoGluon's DataFrame output to numpy array
    # AutoGluon returns a DataFrame with class labels as columns
    y_proba = y_proba_df.values
    
    # --- METRICS ---
    acc = accuracy_score(y_test, y_pred)
    gmean = g_mean_score(y_test, y_pred)
    
    classes = np.arange(n_classes)
    auc = compute_auc(y_test, y_proba, n_classes)
    ce = compute_cross_entropy(y_test, y_proba, classes)
    
    print(f"ACC={acc:.4f} | AUC={auc:.4f} | Time={total_time:.1f}s")
    
    # Cleanup AutoGluon files to save disk space
    try:
        shutil.rmtree(ag_path)
    except Exception:
        pass
    
    del predictor
    
    return {
        "ACC": acc, "AUC_OVO": auc, "G_Mean": gmean, "CE": ce,
        "fit_time_s": fit_time, "predict_time_s": predict_time, "total_time_s": total_time,
    }


# ==============================================================================
# 5. RESULT I/O
# ==============================================================================

CSV_COLUMNS = [
    "task_id", "dataset", "n_samples", "n_features", "n_classes", "regime",
    "model", "ACC", "AUC_OVO", "G_Mean", "CE",
    "fit_time_s", "predict_time_s", "total_time_s",
]


def init_results_csv():
    """Create the CSV file with headers if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_result(row_dict):
    """Append a single result row to the CSV file."""
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row_dict)


def is_already_done(task_id, model_name):
    """Check if this (task_id, model) pair already exists in the CSV (for resumption)."""
    if not os.path.exists(RESULTS_FILE):
        return False
    try:
        df = pd.read_csv(RESULTS_FILE)
        return ((df["task_id"] == task_id) & (df["model"] == model_name)).any()
    except Exception:
        return False


# ==============================================================================
# 6. MAIN LOOP
# ==============================================================================

def main():
    init_results_csv()
    
    total_start = time.time()
    n_datasets = len(datasets_to_run)
    
    # --- Save dataset metadata ---
    pd.DataFrame(DATASETS).to_csv(METADATA_FILE, index=False)
    print(f"📊 Dataset metadata saved to {METADATA_FILE}")
    
    for idx, ds_info in enumerate(datasets_to_run):
        tid = ds_info["tid"]
        print(f"\n{'='*70}")
        print(f"📁 [{idx+1}/{n_datasets}] {ds_info['name']} (tid={tid}, n={ds_info['n']}, "
              f"c={ds_info['classes']}, regime={ds_info['regime']})")
        print(f"{'='*70}")
        
        try:
            # --- Load ---
            X, y, cat_indicator, attr_names, full_name = load_openml_dataset(tid)
            X, y, label_encoder = preprocess(X, y, cat_indicator)
            n_classes = len(label_encoder.classes_)
            
            print(f"  ✅ Loaded: {X.shape[0]} samples, {X.shape[1]} features, {n_classes} classes")
            
            # --- 70/30 stratified split ---
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30, random_state=SEED, stratify=y
            )
            print(f"  📐 Split: train={len(y_train)}, test={len(y_test)}")
            
            # --- Evaluate each model ---
            for model_name, (builder, is_autogluon) in MODELS.items():
                
                # Skip if already computed (allows resumption)
                if is_already_done(tid, model_name):
                    print(f"    ⏭️  {model_name} — already done, skipping.")
                    continue
                
                safe_cleanup()
                
                try:
                    if is_autogluon:
                        metrics = evaluate_autogluon(
                            X_train, X_test, y_train, y_test,
                            n_classes, ds_info["name"]
                        )
                    else:
                        model = builder()
                        metrics = evaluate_sklearn_model(
                            model_name, model, X_train, X_test, y_train, y_test, n_classes
                        )
                        del model  # free memory
                    
                    # Build the result row with EXPLICIT column ordering
                    row = {
                        "task_id":       tid,
                        "dataset":       ds_info["name"],
                        "n_samples":     ds_info["n"],
                        "n_features":    ds_info["features"],
                        "n_classes":     ds_info["classes"],
                        "regime":        ds_info["regime"],
                        "model":         model_name,
                        "ACC":           round(metrics["ACC"], 6),
                        "AUC_OVO":       round(metrics["AUC_OVO"], 6) if not np.isnan(metrics["AUC_OVO"]) else "",
                        "G_Mean":        round(metrics["G_Mean"], 6),
                        "CE":            round(metrics["CE"], 6) if not np.isnan(metrics["CE"]) else "",
                        "fit_time_s":    round(metrics["fit_time_s"], 2),
                        "predict_time_s":round(metrics["predict_time_s"], 2),
                        "total_time_s":  round(metrics["total_time_s"], 2),
                    }
                    append_result(row)
                    
                except Exception as e:
                    print(f"    ❌ {model_name} FAILED: {e}")
                    # Write a failure row so we know it was attempted
                    row = {col: "" for col in CSV_COLUMNS}
                    row["task_id"] = tid
                    row["dataset"] = ds_info["name"]
                    row["model"] = model_name
                    row["ACC"] = "FAILED"
                    row["AUC_OVO"] = str(e)[:100]
                    append_result(row)
                
                safe_cleanup()
            
        except Exception as e:
            print(f"  ❌ DATASET FAILED: {e}")
            safe_cleanup()
    
    elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"🎉 ALL EXPERIMENTS FINISHED in {elapsed/60:.1f} minutes!")
    print(f"📄 Results saved to: {RESULTS_FILE}")
    print(f"📊 Metadata saved to: {METADATA_FILE}")
    print(f"{'='*70}")
    
    # Show a preview of results
    try:
        df = pd.read_csv(RESULTS_FILE)
        print(f"\nTotal rows: {len(df)}")
        print(df.groupby("model")["ACC"].mean().sort_values(ascending=False))
    except Exception:
        pass


# --- RUN ---
main()
```

---

## Cell 3 — Quick Results Preview (Optional)

```python
import pandas as pd

df = pd.read_csv("/kaggle/working/kaggle_results.csv")
print(f"Total results: {len(df)} rows")

# Filter out failed runs before computing averages
df_success = df[df["ACC"] != "FAILED"].copy()

if len(df_success) == 0:
    print("⚠️ All runs failed. No metrics to average. Check the CSV for error messages.")
else:
    print(f"Datasets: {df_success['dataset'].nunique()}")
    print(f"Models: {df_success['model'].nunique()}")
    print()
    
    # Convert metric columns to numeric to avoid TypeErrors
    for col in ["ACC", "AUC_OVO", "G_Mean", "total_time_s"]:
        df_success[col] = pd.to_numeric(df_success[col], errors='coerce')
    
    # Average metrics per model
    summary = df_success.groupby("model").agg({
        "ACC": "mean",
        "AUC_OVO": "mean",
        "G_Mean": "mean",
        "total_time_s": "sum"
    }).round(4).sort_values("AUC_OVO", ascending=False)
    
    print("=== Average Metrics by Model ===")
    print(summary)
```

---

## After Kaggle Finishes

1. Download **both** files from Kaggle's Output pane:
   - `kaggle_results.csv` — the main results
   - `dataset_metadata.csv` — dataset metadata for regime analysis
2. Upload or paste them back into our conversation.
3. I will then generate:
   - Statistical tests (Friedman + Nemenyi CD diagram, Bayesian signed-rank with ROPE)
   - Regime analysis (by size, by # classes, by categorical share, by missing values)
   - Model Card for TabICL v2
   - Report tables and figures

---

## Bugs Fixed from v1

| # | Bug | Impact |
|---|---|---|
| 1 | `TabICLClassifier(random_state=seed)` → must be `seed=seed` | **Crash** on first run |
| 2 | Used raw `lgb.LGBMClassifier` instead of `LGBM_TD_Classifier` | Wrong baselines (not meta-tuned) |
| 3 | `roc_auc_score(multi_class='ovo')` on binary data | **Crash** on binary datasets |
| 4 | AutoGluon `predict(test_data)` with target column included | Messy (AutoGluon ignores it, but bad practice) |
| 5 | No unique `path=` for AutoGluon | Overwrites previous models |
| 6 | No AutoGluon disk cleanup (`shutil.rmtree`) | Fills disk, crashes Kaggle |
| 7 | CSV column order mismatch (dict keys ≠ header) | **Silently scrambled data** |
| 8 | Target `y` never label-encoded | **Crash** on string labels (XGBoost) |
| 9 | G-Mean used confusion matrix instead of per-class recall | Wrong metric values |
| 10 | No `del model` after evaluation | GPU OOM on large datasets |
| 11 | No resumption support | Lost all progress on crash |
| 12 | Dataset metadata not saved | Cannot do regime analysis later |
| 13 | Only `total_time` recorded, not `fit_time` + `predict_time` | Missing required metric |
| 14 | Placeholder dataset IDs (28/30 rejected by TabArena) | Invalid experiment |
| 15 | `anneal` dataset has 100% NaN columns | PyTabKit crashed because `median()` couldn't fill them |
| 16 | `anneal` dataset has zero-variance (constant) columns | TabICL crashed with `boolean index` axis mismatch |
