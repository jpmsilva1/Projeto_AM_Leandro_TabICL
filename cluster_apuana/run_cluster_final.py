import os
import sys
import time
import signal
import warnings
import multiprocessing
import numpy as np
import pandas as pd
import openml
import optuna
from optuna.samplers import TPESampler

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
import torch

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Limita paralelismo para evitar deadlock
num_cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
NUM_CPUS = int(num_cpus_env) if num_cpus_env else multiprocessing.cpu_count()
print(f"Detectado limite de CPUs: {NUM_CPUS}")

os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CPUS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_CPUS)

RESULTS_FILE = "final_run_results.csv"
METADATA_FILE = "final_run_metadata.csv"

# Timeout de segurança
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Tempo limite de segurança atingido!")

RESULTS_FILE = "final_run_results.csv"

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
    
    for i, col in enumerate(X.columns):
        if categorical_indicator[i] or X[col].dtype == "object" or str(X[col].dtype) == "category":
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    
    X = X.astype(float)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median()).fillna(0)
    X = X.loc[:, (X != X.iloc[0]).any()]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_clean)
    return X, y_encoded, le

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
        present_classes = np.unique(y_true)
        if len(present_classes) < 2:
            return np.nan
            
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            filtered_probs = y_proba[:, present_classes]
            row_sums = filtered_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            filtered_probs = filtered_probs / row_sums
            return float(roc_auc_score(y_true, filtered_probs, multi_class="ovo", labels=present_classes))
    except: return np.nan

def compute_cross_entropy(y_true, y_proba, classes):
    try: return float(log_loss(y_true, y_proba, labels=classes))
    except: return np.nan

# --- BASELINES DEFAULT ---
def run_baseline(model_name, build_fn, X_train, y_train, X_test, y_test, n_classes, **kwargs):
    print(f"    🔧 {model_name}...", end="", flush=True)
    t0 = time.perf_counter()
    model = build_fn(**kwargs)
    
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    else:
        # For TabICL
        preds, probs = model(X_train, y_train, X_test)
        
    fit_predict_time = time.perf_counter() - t0
    acc = accuracy_score(y_test, preds)
    gmean = g_mean_score(y_test, preds)
    auc = compute_auc(y_test, probs, n_classes) if probs is not None else np.nan
    ce = compute_cross_entropy(y_test, probs, np.arange(n_classes)) if probs is not None else np.nan
    
    print(f" ACC={acc:.4f} | AUC={auc:.4f} | Tempo={fit_predict_time:.1f}s")
    return {
        "model": model_name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
        "G_Mean": round(gmean, 6), "CE": round(ce, 6),
        "total_time_s": round(fit_predict_time, 2)
    }

def build_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, verbose=-1)

def build_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss', verbosity=0, tree_method='hist', device='cuda')

def build_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=SEED, verbose=0, task_type='GPU')

def build_tabicl():
    from tabicl import TabICLClassifier
    return TabICLClassifier(device="cuda" if torch.cuda.is_available() else "cpu", random_state=SEED)

# --- AUTOGLUON ---
def run_autogluon(preset, X_train, y_train, X_test, y_test, n_classes):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil
    
    name = "AutoGluon_Extreme" if preset == "best_quality" else "AutoGluon_Default"
    time_limit = 3600 if preset == "best_quality" else 1800  # 1h Extreme, 30min Default
    alarm_limit = 5400 if preset == "best_quality" else 2700  # Alarme de segurança
    print(f"    ⚙️ {name} (limite={time_limit//60}min)...", end="", flush=True)
    sys.stdout.flush()
    
    ag_path = tempfile.mkdtemp()
    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test)
    
    t0 = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(alarm_limit)
        
        ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
        ag_problem_type = "binary" if n_classes == 2 else "multiclass"
        predictor = TabularPredictor(label="target", eval_metric=ag_metric, problem_type=ag_problem_type, path=ag_path, verbosity=0)
        predictor.fit(
            train_data=train_df, presets=preset, time_limit=time_limit,
            num_gpus=1, num_cpus=NUM_CPUS, ag_args_fit={"num_cpus": NUM_CPUS}
        )
        
        signal.alarm(0)
        
        y_pred = predictor.predict(test_df).values
        y_proba = predictor.predict_proba(test_df).values
        
        total_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, y_pred)
        gmean = g_mean_score(y_test, y_pred)
        auc = compute_auc(y_test, y_proba, n_classes)
        ce = compute_cross_entropy(y_test, y_proba, np.arange(n_classes))
        
        print(f" ACC={acc:.4f} | AUC={auc:.4f} | Tempo={total_time:.1f}s")
        shutil.rmtree(ag_path, ignore_errors=True)
        
        return {
            "model": name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
            "G_Mean": round(gmean, 6), "CE": round(ce, 6),
            "total_time_s": round(total_time, 2)
        }
    except (TimeoutError, Exception) as e:
        signal.alarm(0)
        shutil.rmtree(ag_path, ignore_errors=True)
        raise e

# --- OPTUNA TUNING ---
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def cross_val_objective(model_class, params, X, y):
    try:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        splits = list(cv.split(X, y))
    except ValueError:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
        splits = list(cv.split(X))
        
    scores = []
    for train_idx, val_idx in splits:
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]
        
        le_fold = LabelEncoder()
        y_tr_enc = le_fold.fit_transform(y_tr)
        
        model = model_class(**params)
        model.fit(X_tr, y_tr_enc)
        
        y_pred_enc = model.predict(X_va)
        y_pred = le_fold.inverse_transform(y_pred_enc)
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
        'verbosity': 0,
        'tree_method': 'hist',
        'device': 'cuda'
    }
    return cross_val_objective(XGBClassifier, params, X, y)

def objective_cat(trial, X, y):
    params = {
        'iterations': trial.suggest_int('iterations', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'depth': trial.suggest_int('depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'bootstrap_type': 'Bernoulli',
        'random_state': SEED,
        'verbose': 0,
        'task_type': 'GPU'
    }
    return cross_val_objective(CatBoostClassifier, params, X, y)

def run_tuning(model_name, objective_func, model_class, X_train, y_train, X_test, y_test, n_classes):
    print(f"    ⚙️ Tuning {model_name}...", end="", flush=True)
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
        best_params['tree_method'] = 'hist'
        best_params['device'] = 'cuda'
    elif model_name == "CatBoost_Tuned": 
        best_params['verbose'] = 0
        best_params['task_type'] = 'GPU'
        best_params['bootstrap_type'] = 'Bernoulli'
    
    best_model = model_class(**best_params)
    
    fit_start = time.perf_counter()
    le_final = LabelEncoder()
    y_train_enc = le_final.fit_transform(y_train)
    best_model.fit(X_train, y_train_enc)
    fit_time = time.perf_counter() - fit_start
    
    pred_start = time.perf_counter()
    y_pred_enc = best_model.predict(X_test)
    y_pred = le_final.inverse_transform(y_pred_enc)
    
    y_proba_enc = best_model.predict_proba(X_test) if hasattr(best_model, "predict_proba") else None
    y_proba = None
    if y_proba_enc is not None:
        y_proba = np.zeros((len(X_test), n_classes))
        for i, cls in enumerate(le_final.classes_):
            if cls < n_classes:
                y_proba[:, cls] = y_proba_enc[:, i]
    pred_time = time.perf_counter() - pred_start
    acc = accuracy_score(y_test, y_pred)
    gmean = g_mean_score(y_test, y_pred)
    auc = compute_auc(y_test, y_proba, n_classes) if y_proba is not None else np.nan
    ce = compute_cross_entropy(y_test, y_proba, np.arange(n_classes)) if y_proba is not None else np.nan
    
    total_time = time.perf_counter() - t0
    print(f" ACC={acc:.4f} | AUC={auc:.4f} | Tempo={total_time:.1f}s")
    return {
        "model": model_name, "ACC": round(acc, 6), "AUC_OVO": round(auc, 6), 
        "G_Mean": round(gmean, 6), "CE": round(ce, 6),
        "total_time_s": round(total_time, 2)
    }

# --- MAIN ---
def main():
    print("🔥 INICIANDO PIPELINE MONOLITICO OTIMIZADO PARA RUNPOD (GPU ACCELERATION) 🔥")
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
        
        models_to_run = [
            "TabICL", "LightGBM_TD", "XGBoost_TD", "CatBoost_TD",
            "AutoGluon_Default", "AutoGluon_Extreme",
            "LightGBM_Tuned", "XGBoost_Tuned", "CatBoost_Tuned"
        ]
        
        if all((ds['name'], m) in done_pairs for m in models_to_run):
            print("  ✅ All models complete, skipping...")
            continue
            
        # Correção Crítica: O ID fornecido no dicionário é o Task ID (tid). 
        # Não podemos usar get_dataset(tid) diretamente pois pode haver colisão de IDs no OpenML 
        # (ex: carregar um dataset de imagens que acidentalmente tem o mesmo ID da nossa task).
        # Devemos SEMPRE pegar a task primeiro para descobrir o Dataset ID correto.
        try:
            try:
                task = openml.tasks.get_task(ds["tid"])
                dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=False, download_features_meta_data=True)
            except openml.exceptions.OpenMLServerException as e:
                if "Unknown task" in str(e):
                    print(f"    ⚠️ Task {ds['tid']} inexistente. Tentando como Dataset ID direto...")
                    dataset = openml.datasets.get_dataset(ds["tid"], download_data=True, download_qualities=False, download_features_meta_data=True)
                else:
                    raise e

            X, y, cat_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
            n_classes = len(np.unique(y.dropna()))
            X_clean, y_clean, le = preprocess(X, y, cat_indicator)
            
            try: X_train, X_test, y_train, y_test = train_test_split(X_clean.values, y_clean, test_size=0.3, random_state=SEED, stratify=y_clean)
            except ValueError: X_train, X_test, y_train, y_test = train_test_split(X_clean.values, y_clean, test_size=0.3, random_state=SEED)
            
            # --- EXECUÇÃO DOS MODELOS ---
            def execute_model(m_name, run_func, *args):
                if (ds['name'], m_name) not in done_pairs:
                    try:
                        res = run_func(*args)
                        res['dataset'] = ds['name']
                        res['regime'] = ds['regime']
                        all_results.append(res)
                        pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
                    except Exception as e:
                        print(f"    ❌ {m_name} FAILED: {e}")

            # Defaults
            execute_model("TabICL", run_baseline, "TabICL", build_tabicl, X_train, y_train, X_test, y_test, n_classes)
            execute_model("LightGBM_TD", run_baseline, "LightGBM_TD", build_lightgbm, X_train, y_train, X_test, y_test, n_classes)
            execute_model("XGBoost_TD", run_baseline, "XGBoost_TD", build_xgboost, X_train, y_train, X_test, y_test, n_classes)
            execute_model("CatBoost_TD", run_baseline, "CatBoost_TD", build_catboost, X_train, y_train, X_test, y_test, n_classes)
            
            # AutoGluon
            execute_model("AutoGluon_Default", run_autogluon, "medium_quality", X_train, y_train, X_test, y_test, n_classes)
            execute_model("AutoGluon_Extreme", run_autogluon, "best_quality", X_train, y_train, X_test, y_test, n_classes)
            
            # Tuned
            execute_model("LightGBM_Tuned", run_tuning, "LightGBM_Tuned", objective_lgb, LGBMClassifier, X_train, y_train, X_test, y_test, n_classes)
            execute_model("XGBoost_Tuned", run_tuning, "XGBoost_Tuned", objective_xgb, XGBClassifier, X_train, y_train, X_test, y_test, n_classes)
            execute_model("CatBoost_Tuned", run_tuning, "CatBoost_Tuned", objective_cat, CatBoostClassifier, X_train, y_train, X_test, y_test, n_classes)
                    
        except Exception as e: print(f"  ❌ DATASET FAILED: {e}")

if __name__ == "__main__":
    main()
