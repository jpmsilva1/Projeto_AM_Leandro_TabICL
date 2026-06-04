import os
import sys
import time
import signal
import shutil
import tempfile
import warnings
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor

# --- CONFIGURAÇÕES DO CLUSTER ---
openml.config.cache_directory = os.path.expanduser('./openml_cache')
warnings.filterwarnings("ignore")

# Limita paralelismo para evitar deadlock no SLURM
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

RESULTS_FILE = "cluster_results.csv"
METADATA_FILE = "dataset_metadata.csv"

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
DEVICE = "cuda"

# --- TIMEOUT DE SEGURANÇA ---
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Tempo limite de segurança atingido!")

# --- FUNÇÕES DE PRE-PROCESSAMENTO ---
def preprocess(X, y, categorical_indicator):
    X = X.copy()
    
    # Codifica colunas categóricas
    for i, col in enumerate(X.columns):
        if categorical_indicator[i] or X[col].dtype == "object" or str(X[col].dtype) == "category":
            X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)
    
    X = X.astype(float)
    X = X.dropna(axis=1, how='all')
    X = X.fillna(X.median()).fillna(0)
    X = X.loc[:, (X != X.iloc[0]).any()]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(np.asarray(y).ravel())
    return X, y_encoded, le

def safe_run(model_name, build_fn, X_train, y_train, X_test, y_test, n_classes, **kwargs):
    print(f"    🔧 {model_name}...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        model = build_fn(**kwargs)
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)
        else:
            preds, probs = model(X_train, y_train, X_test)
            
        fit_predict_time = time.perf_counter() - t0
        
        acc = accuracy_score(y_test, preds)
        if n_classes == 2:
            auc = roc_auc_score(y_test, probs[:, 1])
        else:
            auc = roc_auc_score(y_test, probs, multi_class="ovo", average="macro")
            
        print(f" ACC={acc:.4f} | AUC={auc:.4f} | Time={fit_predict_time:.1f}s")
        return {"ACC": acc, "AUC_OVO": auc, "total_time_s": fit_predict_time}
    except Exception as e:
        print(f" ❌ {model_name} FAILED: {str(e)[:150]}")
        return {"ACC": "FAILED", "AUC_OVO": str(e)[:100], "total_time_s": 0}

# --- MODELOS ---
def build_tabicl():
    from tabicl import TabICLClassifier
    return TabICLClassifier(device=DEVICE)

def build_lightgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=SEED, verbose=-1)

def build_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric='logloss', verbosity=0)

def build_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier(random_state=SEED, verbose=0)

# --- RETOMADA INTELIGENTE ---
def load_done_pairs():
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        return set(zip(df['dataset'], df['model'])), df.to_dict('records')
    return set(), []

# --- PIPELINE PRINCIPAL ---
print("=================================================================")
print("🚀 INICIANDO PIPELINE NO CLUSTER APUANA (COM RETOMADA INTELIGENTE)")
print("=================================================================")
sys.stdout.flush()

done_pairs, all_results = load_done_pairs()
dataset_meta = []

for i, ds in enumerate(DATASETS):
    print(f"\n=================================================================")
    print(f"📁 [{i+1}/{len(DATASETS)}] {ds['name']} (tid={ds['tid']}, regime={ds['regime']})")
    print(f"=================================================================")
    sys.stdout.flush()
    
    # Verifica quais modelos já foram feitos para este dataset
    models_needed = ["TabICL v2", "LightGBM_TD", "XGBoost_TD", "CatBoost_TD", "AutoGluon"]
    if all((ds['name'], m) in done_pairs for m in models_needed):
        print("  ✅ Todos os modelos já completos, pulando...")
        sys.stdout.flush()
        continue
    
    try:
        try:
            dataset = openml.datasets.get_dataset(ds["tid"], download_data=True, download_qualities=False, download_features_meta_data=True)
        except Exception:
            task = openml.tasks.get_task(ds["tid"])
            dataset = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=False, download_features_meta_data=True)
        X, y, cat_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
        n_classes = len(np.unique(y.dropna()))
        
        dataset_meta.append({"dataset": ds["name"], "regime": ds["regime"], "n_samples": len(X), "n_features": len(X.columns), "n_classes": n_classes})
        pd.DataFrame(dataset_meta).to_csv(METADATA_FILE, index=False)
        
        print(f"✅ Loaded: {len(X)} samples, {len(X.columns)} features, {n_classes} classes")
        sys.stdout.flush()
        
        X_clean, y_clean, le = preprocess(X, y, cat_indicator)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=SEED, stratify=y_clean)
        except ValueError:
            print("  ⚠️ Stratified split failed (rare class), falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=SEED)
        
        def save_result(model_name, result):
            all_results.append({"dataset": ds["name"], "model": model_name, **result})
            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
            sys.stdout.flush()
        
        # 1. TabICL
        if (ds['name'], "TabICL v2") not in done_pairs:
            res = safe_run("TabICL", build_tabicl, X_train.values, y_train, X_test.values, y_test, n_classes)
            save_result("TabICL v2", res)
        
        # 2. LightGBM
        if (ds['name'], "LightGBM_TD") not in done_pairs:
            res = safe_run("LightGBM_TD", build_lightgbm, X_train.values, y_train, X_test.values, y_test, n_classes)
            save_result("LightGBM_TD", res)
        
        # 3. XGBoost
        if (ds['name'], "XGBoost_TD") not in done_pairs:
            res = safe_run("XGBoost_TD", build_xgboost, X_train.values, y_train, X_test.values, y_test, n_classes)
            save_result("XGBoost_TD", res)
        
        # 4. CatBoost
        if (ds['name'], "CatBoost_TD") not in done_pairs:
            res = safe_run("CatBoost_TD", build_catboost, X_train.values, y_train, X_test.values, y_test, n_classes)
            save_result("CatBoost_TD", res)
        
        # 5. AutoGluon (BEST QUALITY) — com proteção anti-deadlock
        if (ds['name'], "AutoGluon") not in done_pairs:
            print(f"    🔧 AutoGluon (Best)...", end="", flush=True)
            t0 = time.perf_counter()
            
            ag_path = tempfile.mkdtemp()
            ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
            
            train_df = X_train.copy()
            train_df["target"] = y_train
            test_df = X_test.copy()
            
            try:
                # Timeout de segurança de 2 horas via signal (funciona no Linux/SLURM)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(7200)  # 2 horas em segundos
                
                ag_problem_type = "binary" if n_classes == 2 else "multiclass"
                predictor = TabularPredictor(
                    label="target", eval_metric=ag_metric, problem_type=ag_problem_type, path=ag_path, verbosity=0
                )
                predictor.fit(
                    train_data=train_df,
                    presets="best_quality",
                    time_limit=3600,  # 1 hora de limite interno do AutoGluon
                    num_cpus=4,       # Limita CPUs para evitar deadlock no SLURM
                    ag_args_fit={"num_cpus": 4},  # Limita CPUs nos modelos internos
                )
                
                signal.alarm(0)  # Desliga o alarme se tudo correu bem
                
                fit_time = time.perf_counter() - t0
                
                preds = predictor.predict(test_df)
                probs = predictor.predict_proba(test_df)
                probs = probs.reindex(sorted(probs.columns), axis=1)
                if n_classes == 2:
                    auc = roc_auc_score(y_test, probs.iloc[:, 1])
                else:
                    auc = roc_auc_score(y_test, probs, multi_class="ovo", average="macro")
                acc = accuracy_score(y_test, preds)
                
                print(f" ACC={acc:.4f} | AUC={auc:.4f} | Tempo Total={fit_time:.1f}s")
                save_result("AutoGluon", {"ACC": acc, "AUC_OVO": auc, "total_time_s": fit_time})
            except (TimeoutError, Exception) as e:
                signal.alarm(0)
                print(f" ❌ AutoGluon FAILED: {str(e)[:150]}")
                save_result("AutoGluon", {"ACC": "FAILED", "AUC_OVO": str(e)[:100], "total_time_s": 0})
            finally:
                shutil.rmtree(ag_path, ignore_errors=True)
            
    except Exception as e:
        print(f" ❌ DATASET FAILED: {e}")
    
    sys.stdout.flush()

print("=================================================================")
print("🎉 TODOS OS EXPERIMENTOS FORAM CONCLUÍDOS!")
print(f"Resultados salvos em: {RESULTS_FILE}")
print("=================================================================")

# Gera o sumário final
df = pd.read_csv(RESULTS_FILE)
df_success = df[df["ACC"] != "FAILED"].copy()

if len(df_success) > 0:
    for col in ["ACC", "AUC_OVO", "total_time_s"]:
        df_success[col] = pd.to_numeric(df_success[col], errors='coerce')
    
    summary = df_success.groupby("model").agg({
        "ACC": "mean",
        "AUC_OVO": "mean",
        "total_time_s": "sum"
    }).round(4).sort_values("AUC_OVO", ascending=False)
    
    print("\n=== MÉDIA GERAL (POR MODELO) ===")
    print(summary)
