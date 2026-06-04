import os
import time
import warnings
import numpy as np
import pandas as pd
import openml
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_FILE = "optuna_results.csv"

# --- DATASETS OFICIAIS ---
DATASETS = [
    {'tid': 1464, 'name': 'blood-transfusion-service-center', 'regime': 'small'},
    {'tid': 37, 'name': 'diabetes', 'regime': 'small'},
    {'tid': 2, 'name': 'anneal', 'regime': 'small'},
    {'tid': 168757, 'name': 'credit-g', 'regime': 'medium'},
    {'tid': 359956, 'name': 'qsar-biodeg', 'regime': 'medium'},
    {'tid': 2077, 'name': 'baseball', 'regime': 'medium'},
    {'tid': 2073, 'name': 'yeast', 'regime': 'medium'},
    {'tid': 45, 'name': 'splice', 'regime': 'medium'},
    {'tid': 359967, 'name': 'Bioresponse', 'regime': 'medium'},
    {'tid': 3011, 'name': 'hypothyroid', 'regime': 'medium'},
    {'tid': 3892, 'name': 'hiva_agnostic', 'regime': 'medium'},
    {'tid': 43, 'name': 'spambase', 'regime': 'medium'},
    {'tid': 58, 'name': 'waveform-5000', 'regime': 'medium'},
    {'tid': 359968, 'name': 'churn', 'regime': 'medium'},
    {'tid': 30, 'name': 'page-blocks', 'regime': 'medium'},
    {'tid': 28, 'name': 'optdigits', 'regime': 'medium'},
    {'tid': 2074, 'name': 'satimage', 'regime': 'medium'},
    {'tid': 3481, 'name': 'isolet', 'regime': 'medium'},
    {'tid': 24, 'name': 'mushroom', 'regime': 'medium'},
    {'tid': 3510, 'name': 'JapaneseVowels', 'regime': 'medium'},
    {'tid': 32, 'name': 'pendigits', 'regime': 'large'},
    {'tid': 26, 'name': 'nursery', 'regime': 'large'},
    {'tid': 6, 'name': 'letter', 'regime': 'large'},
    {'tid': 3688, 'name': 'houses', 'regime': 'large'},
    {'tid': 359979, 'name': 'Amazon_employee_access', 'regime': 'large'},
    {'tid': 3945, 'name': 'KDDCup09_appetency', 'regime': 'large'},
    {'tid': 168868, 'name': 'APSFailure', 'regime': 'large'},
    {'tid': 361329, 'name': 'KDD98', 'regime': 'large'},
    {'tid': 211986, 'name': 'Diabetes130US', 'regime': 'large'},
    {'tid': 360113, 'name': 'porto-seguro', 'regime': 'large'},
]

SEED = 42
N_TRIALS = 50

# --- PRE-PROCESSAMENTO ---
def preprocess(X, y, categorical_indicator):
    X = X.copy()
    y_clean = y.dropna()
    X = X.loc[y_clean.index]
    
    numeric_transformer = SimpleImputer(strategy='mean')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    numeric_cols = [i for i, is_cat in enumerate(categorical_indicator) if not is_cat]
    categorical_cols = [i for i, is_cat in enumerate(categorical_indicator) if is_cat]
    
    X_proc = pd.DataFrame(index=X.index, columns=X.columns)
    if numeric_cols:
        X_proc.iloc[:, numeric_cols] = numeric_transformer.fit_transform(X.iloc[:, numeric_cols])
    if categorical_cols:
        X_proc.iloc[:, categorical_cols] = categorical_transformer.fit_transform(X.iloc[:, categorical_cols])
        
    X_proc = X_proc.fillna(0)
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_clean)
    return X_proc, y_encoded, le

def g_mean_score(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if not mask.any(): continue
        recalls.append(float((y_pred[mask] == c).mean()))
    if not recalls: return 0.0
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0)))))

def compute_auc(y_true, y_proba, n_classes):
    try:
        if n_classes == 2: return float(roc_auc_score(y_true, y_proba[:, 1]))
        else: return float(roc_auc_score(y_true, y_proba, multi_class="ovo"))
    except: return np.nan

def compute_cross_entropy(y_true, y_proba, classes):
    try: return float(log_loss(y_true, y_proba, labels=classes))
    except: return np.nan

# --- OPTUNA OBJECTIVES COM CROSS-VALIDATION ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def cross_val_objective(model_class, params, X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        model = model_class(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        scores.append(accuracy_score(y_va, y_pred))
    return np.mean(scores)

def objective_lgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': SEED,
        'verbose': -1
    }
    return cross_val_objective(LGBMClassifier, params, X, y)

def objective_xgb(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': SEED,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'verbosity': 0
    }
    return cross_val_objective(XGBClassifier, params, X, y)

def objective_cat(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_state': SEED,
        'verbose': 0
    }
    return cross_val_objective(CatBoostClassifier, params, X, y)

# --- RUNNER ---
def run_tuning(model_name, objective_func, model_class, params_key, X_train, y_train, X_test, y_test, n_classes):
    print(f"    ⚙️ Tuning {model_name} ({N_TRIALS} trials)...", end="", flush=True)
    t0 = time.perf_counter()
        
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=SEED))
    study.optimize(lambda trial: objective_func(trial, X_train, y_train), n_trials=N_TRIALS)
    
    best_params = study.best_params
    best_params['random_state'] = SEED
    if model_name == "LightGBM_Tuned": best_params['verbose'] = -1
    elif model_name == "XGBoost_Tuned": 
        best_params['use_label_encoder'] = False
        best_params['eval_metric'] = 'logloss'
        best_params['verbosity'] = 0
    elif model_name == "CatBoost_Tuned": best_params['verbose'] = 0
    
    best_model = model_class(**best_params)
    
    fit_start = time.perf_counter()
    best_model.fit(X_train, y_train)
    fit_time = time.perf_counter() - fit_start
    
    pred_start = time.perf_counter()
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
    pred_time = time.perf_counter() - pred_start
    
    total_time = time.perf_counter() - t0
    
    acc = accuracy_score(y_test, y_pred)
    gmean = g_mean_score(y_test, y_pred)
    auc = compute_auc(y_test, y_proba, n_classes) if y_proba is not None else np.nan
    ce = compute_cross_entropy(y_test, y_proba, np.arange(n_classes)) if y_proba is not None else np.nan
    
    print(f" ACC={acc:.4f} | AUC={auc:.4f} | Tempo Total={total_time:.1f}s")
    
    return {
        "model": model_name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
        "G_Mean": round(gmean, 6), "CE": round(ce, 6),
        "fit_time_s": round(fit_time, 2), "predict_time_s": round(pred_time, 2), "total_time_s": round(total_time, 2)
    }

def run_autogluon_default(X_train, y_train, X_test, y_test, n_classes):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil
    print(f"    ⚙️ AutoGluon (Default)...", end="", flush=True)
    ag_path = tempfile.mkdtemp()
    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test)
    
    t0 = time.perf_counter()
    ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
    predictor = TabularPredictor(label="target", eval_metric=ag_metric, path=ag_path, verbosity=0)
    # Default preset (medium_quality) and fast training
    predictor.fit(train_data=train_df, time_limit=1800)
    fit_time = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    y_pred = predictor.predict(test_df).values
    y_proba = predictor.predict_proba(test_df).values
    pred_time = time.perf_counter() - t0
    
    total_time = fit_time + pred_time
    acc = accuracy_score(y_test, y_pred)
    gmean = g_mean_score(y_test, y_pred)
    auc = compute_auc(y_test, y_proba, n_classes)
    ce = compute_cross_entropy(y_test, y_proba, np.arange(n_classes))
    
    print(f" ACC={acc:.4f} | AUC={auc:.4f} | Tempo Total={total_time:.1f}s")
    shutil.rmtree(ag_path, ignore_errors=True)
    
    return {
        "model": "AutoGluon_Default", "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
        "G_Mean": round(gmean, 6), "CE": round(ce, 6),
        "fit_time_s": round(fit_time, 2), "predict_time_s": round(pred_time, 2), "total_time_s": round(total_time, 2)
    }


def main():
    all_results = []
    if os.path.exists(RESULTS_FILE):
        all_results = pd.read_csv(RESULTS_FILE).to_dict('records')
        done_pairs = set((r['dataset'], r['model']) for r in all_results)
    else:
        done_pairs = set()

    for ds in DATASETS:
        print(f"\n{'='*65}")
        print(f"📁 {ds['name']} (tid={ds['tid']}, regime={ds['regime']})")
        print(f"{'='*65}")
        
        models_to_run = ["LightGBM_Tuned", "XGBoost_Tuned", "CatBoost_Tuned", "AutoGluon_Default"]
        if all((ds['name'], m) in done_pairs for m in models_to_run):
            print("  ✅ All models complete, skipping...")
            continue
            
        try:
            try:
                dataset = openml.datasets.get_dataset(ds["tid"], download_data=True, download_qualities=False, download_features_meta_data=True)
            except Exception:
                task = openml.tasks.get_task(ds["tid"])
                dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=False, download_features_meta_data=True)
                
            X, y, cat_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
            n_classes = len(np.unique(y.dropna()))
            X_clean, y_clean, le = preprocess(X, y, cat_indicator)
            
            try: X_train, X_test, y_train, y_test = train_test_split(X_clean.values, y_clean, test_size=0.3, random_state=SEED, stratify=y_clean)
            except ValueError: X_train, X_test, y_train, y_test = train_test_split(X_clean.values, y_clean, test_size=0.3, random_state=SEED)
            
            # AutoGluon Default
            if (ds['name'], "AutoGluon_Default") not in done_pairs:
                try:
                    res = run_autogluon_default(X_train, y_train, X_test, y_test, n_classes)
                    res['dataset'] = ds['name']
                    res['regime'] = ds['regime']
                    all_results.append(res)
                    pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
                except Exception as e: print(f"    ❌ AutoGluon_Default FAILED: {e}")
            
            # Optuna Models
            for m_name, obj_func, m_class in [
                ("LightGBM_Tuned", objective_lgb, LGBMClassifier),
                ("XGBoost_Tuned", objective_xgb, XGBClassifier),
                ("CatBoost_Tuned", objective_cat, CatBoostClassifier)
            ]:
                if (ds['name'], m_name) in done_pairs: continue
                try:
                    res = run_tuning(m_name, obj_func, m_class, None, X_train, y_train, X_test, y_test, n_classes)
                    res['dataset'] = ds['name']
                    res['regime'] = ds['regime']
                    all_results.append(res)
                    pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
                except Exception as e: print(f"    ❌ {m_name} FAILED: {e}")
                    
        except Exception as e: print(f"  ❌ DATASET FAILED: {e}")

if __name__ == "__main__":
    main()
