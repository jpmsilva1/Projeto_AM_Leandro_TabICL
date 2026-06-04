import os
import time
import warnings
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor

# --- CONFIGURAÇÕES DO CLUSTER ---
# Usa uma pasta local no cluster para o cache do OpenML, evitando travamentos
openml.config.cache_directory = os.path.expanduser('./openml_cache')
warnings.filterwarnings("ignore")

RESULTS_FILE = "cluster_results.csv"
METADATA_FILE = "dataset_metadata.csv"

# --- DATASETS OFICIAIS ---
DATASETS = [
    # Small (3)
    {'tid': 359955, 'name': 'blood-transfusion-service-center', 'regime': 'small'},
    {'tid': 37, 'name': 'diabetes', 'regime': 'small'},
    {'tid': 2, 'name': 'anneal', 'regime': 'small'},
    # Medium (17)
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
    # Large (10)
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
        # ROC AUC
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

def build_lightgbm_td():
    from pytabkit.models.sklearn.interface import LGBM_TD_Classifier
    return LGBM_TD_Classifier(random_state=SEED)

def build_xgboost_td():
    from pytabkit.models.sklearn.interface import XGBoost_TD_Classifier
    return XGBoost_TD_Classifier(random_state=SEED)

def build_catboost_td():
    from pytabkit.models.sklearn.interface import CatBoost_TD_Classifier
    return CatBoost_TD_Classifier(random_state=SEED)

# --- PIPELINE PRINCIPAL ---
print("=================================================================")
print("🚀 INICIANDO PIPELINE NO CLUSTER APUANA")
print("=================================================================")

all_results = []
dataset_meta = []

for i, ds in enumerate(DATASETS):
    print(f"\n=================================================================")
    print(f"📁 [{i+1}/{len(DATASETS)}] {ds['name']} (tid={ds['tid']}, regime={ds['regime']})")
    print(f"=================================================================")
    
    try:
        dataset = openml.datasets.get_dataset(ds["tid"], download_data=True, download_qualities=False, download_features_meta_data=True)
        X, y, cat_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)
        n_classes = len(np.unique(y.dropna()))
        
        # Meta log
        dataset_meta.append({"dataset": ds["name"], "regime": ds["regime"], "n_samples": len(X), "n_features": len(X.columns), "n_classes": n_classes})
        pd.DataFrame(dataset_meta).to_csv(METADATA_FILE, index=False)
        
        print(f"✅ Loaded: {len(X)} samples, {len(X.columns)} features, {n_classes} classes")
        
        X_clean, y_clean, le = preprocess(X, y, cat_indicator)
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=SEED, stratify=y_clean)
        except ValueError:
            print("  ⚠️ Stratified split failed (rare class), falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=SEED)
        
        # 1. TabICL
        res_tabicl = safe_run("TabICL", build_tabicl, X_train.values, y_train, X_test.values, y_test, n_classes)
        all_results.append({"dataset": ds["name"], "model": "TabICL v2", **res_tabicl})
        
        # 2. LightGBM
        res_lgbm = safe_run("LightGBM_TD", build_lightgbm_td, X_train.values, y_train, X_test.values, y_test, n_classes)
        all_results.append({"dataset": ds["name"], "model": "LightGBM_TD", **res_lgbm})
        
        # 3. XGBoost
        res_xgb = safe_run("XGBoost_TD", build_xgboost_td, X_train.values, y_train, X_test.values, y_test, n_classes)
        all_results.append({"dataset": ds["name"], "model": "XGBoost_TD", **res_xgb})
        
        # 4. CatBoost
        res_cb = safe_run("CatBoost_TD", build_catboost_td, X_train.values, y_train, X_test.values, y_test, n_classes)
        all_results.append({"dataset": ds["name"], "model": "CatBoost_TD", **res_cb})
        
        # 5. AutoGluon (BEST QUALITY)
        print(f"    🔧 AutoGluon (Best)...", end="", flush=True)
        t0 = time.perf_counter()
        
        ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
        predictor = TabularPredictor(label="target", eval_metric=ag_metric, path=f"./ag_models/{ds['name']}", verbosity=0)
        
        train_df = X_train.copy()
        train_df["target"] = y_train
        test_df = X_test.copy()
        test_df["target"] = y_test
        
        try:
            predictor.fit(
                train_data=train_df,
                presets="best_quality",
                time_limit=14400,  # 4 horas de limite por dataset no modo best_quality
            )
            fit_time = time.perf_counter() - t0
            
            preds = predictor.predict(test_df)
            probs = predictor.predict_proba(test_df)
            probs = probs.reindex(sorted(probs.columns), axis=1) # Prevent silent column ordering bug
            if n_classes == 2:
                auc = roc_auc_score(y_test, probs.iloc[:, 1])
            else:
                auc = roc_auc_score(y_test, probs, multi_class="ovo", average="macro")
            acc = accuracy_score(y_test, preds)
            
            print(f" ACC={acc:.4f} | AUC={auc:.4f} | Time={fit_time:.1f}s")
            all_results.append({"dataset": ds["name"], "model": "AutoGluon", "ACC": acc, "AUC_OVO": auc, "total_time_s": fit_time})
        except Exception as e:
            print(f" ❌ AutoGluon FAILED: {str(e)[:150]}")
            all_results.append({"dataset": ds["name"], "model": "AutoGluon", "ACC": "FAILED", "AUC_OVO": str(e)[:100], "total_time_s": 0})
            
        pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
        
    except Exception as e:
        print(f" ❌ DATASET FAILED: {e}")

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
