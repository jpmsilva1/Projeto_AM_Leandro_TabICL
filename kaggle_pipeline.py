"""
=============================================================================
Pipeline Completo para Kaggle — TabICL v2 + Baselines com Optuna Tuning
=============================================================================

Este script foi projetado para rodar em um Kaggle Notebook com GPU (T4/P100).
Ele realiza:
  1. Instalação automática das dependências necessárias.
  2. Download dos 30 datasets do TabArena-v0.1 via OpenML (com cache seguro).
  3. Busca de hiperparâmetros via Optuna (holdout interno no treino) para TODOS os modelos.
  4. Avaliação final no conjunto de teste (30%) com os melhores hiperparâmetros.
  5. Análise estatística (Friedman/Nemenyi CD diagram + Bayesian ROPE).
  6. Salvamento incremental dos resultados em CSV.

Instruções para o Kaggle:
  - Crie um novo Notebook no Kaggle.
  - Ative o acelerador GPU (Settings > Accelerator > GPU T4 x2).
  - Cole TODO o conteúdo deste arquivo em uma única célula do notebook.
  - Execute a célula. O script cuida de tudo automaticamente.
  - Os resultados serão salvos em /kaggle/working/results/.

Autor: Pipeline aprimorado gerado para o Projeto Final de AM.
"""

# =============================================================================
# 0. Instalação de dependências (roda apenas no Kaggle)
# =============================================================================
import subprocess
import sys
import os

def install_packages():
    """Instala pacotes necessários que não vêm pré-instalados no Kaggle."""
    packages = [
        "openml",
        "tabicl>=2.0,<3.0",
        "optuna>=3.6,<5.0",
        "autorank>=1.2,<2.0",
        "baycomp>=1.0,<2.0",
        "catboost>=1.2,<2.0",
        "xgboost>=2.0,<4.0",
        "lightgbm>=4.3,<5.0",
    ]
    print("[SETUP] Instalando todas as dependências (isso pode levar alguns minutos)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q"] + packages,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("[SETUP] AVISO: Algumas dependências falharam. Tentando uma a uma...")
        print(result.stderr[-500:] if result.stderr else "Sem detalhes do erro.")
        for pkg in packages:
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "-q", pkg],
                    stderr=subprocess.PIPE,
                )
                print(f"  ✅ {pkg}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ {pkg} — erro: {e}")
    else:
        print("[SETUP] Todas as dependências instaladas com sucesso!")
    print()

IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    install_packages()

# =============================================================================
# 1. Imports e Configurações Globais
# =============================================================================
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import optuna
import openml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

if IS_KAGGLE:
    # Configurar cache do OpenML para não lotar a partição root do Kaggle
    openml.config.set_root_cache_directory("/kaggle/working/openml_cache")

SEED = 42
TEST_SIZE = 0.30
OUTPUT_DIR = Path("/kaggle/working/results") if IS_KAGGLE else Path("results/kaggle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV = OUTPUT_DIR / "raw_tuned.csv"
BEST_PARAMS_CSV = OUTPUT_DIR / "best_params.csv"
BAYESIAN_CSV = OUTPUT_DIR / "bayesian_rope.csv"
CD_DIAGRAM_PNG = OUTPUT_DIR / "cd_diagram.png"

# Optuna config (Trials balanceados pelo custo computacional)
N_TRIALS_TABICL = 15      # TabICL é mais pesado
N_TRIALS_BASELINES = 30   # Baselines são mais rápidos (e rodam na GPU)
SCORING = "roc_auc_ovo"

# Dispositivo principal
import torch
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Output: {OUTPUT_DIR}")

# 30 datasets do TabArena-v0.1
RECOMMENDED_TASK_IDS = [
    37, 359955, 15, 49, 3484,
    2, 11, 41, 3022, 3512,
    3892, 168757, 359956, 359967, 359968,
    45, 233214, 233215, 359935, 360932,
    3688, 3945, 168868, 359979, 360113,
    211986, 233211, 233212, 6, 26,
]


# =============================================================================
# 2. Data Loading & Preprocessing
# =============================================================================
@dataclass(frozen=True)
class TabularDataset:
    task_id: int
    name: str
    X: pd.DataFrame
    y: np.ndarray

    @property
    def n_samples(self) -> int: return self.X.shape[0]
    @property
    def n_features(self) -> int: return self.X.shape[1]
    @property
    def n_classes(self) -> int: return int(np.unique(self.y).size)

def load_task(task_id: int) -> TabularDataset:
    task = openml.tasks.get_task(task_id, download_data=True)
    dataset = task.get_dataset()
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    return TabularDataset(task_id=task_id, name=dataset.name, X=X, y=np.asarray(y))

def preprocess(X_train: pd.DataFrame, X_test: pd.DataFrame):
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    X_tr, X_te = X_train.copy(), X_test.copy()

    # Label Encoding
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_tr[col], X_te[col]], axis=0).astype(str)
        le.fit(combined)
        X_tr[col] = le.transform(X_tr[col].astype(str))
        X_te[col] = le.transform(X_te[col].astype(str))

    # NaN Imputation
    if X_tr.isna().any().any() or X_te.isna().any().any():
        imputer = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns, index=X_te.index)

    return X_tr, X_te


# =============================================================================
# 3. Evaluation Metrics
# =============================================================================
def g_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = np.unique(y_true)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in classes if (y_true == c).any()]
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0))))) if recalls else 0.0

def evaluate(estimator, X_train, y_train, X_test, y_test) -> dict:
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
    predict_time = time.perf_counter() - t0

    classes = np.unique(np.concatenate([y_train, y_test]))
    if y_proba is None:
        auc, ce = float("nan"), float("nan")
    elif classes.size == 2:
        auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        ce = float(log_loss(y_test, y_proba, labels=classes))
    else:
        auc = float(roc_auc_score(y_test, y_proba, multi_class="ovo", labels=classes))
        ce = float(log_loss(y_test, y_proba, labels=classes))

    return {
        "auc_ovo": auc, "accuracy": float(accuracy_score(y_test, y_pred)),
        "g_mean": g_mean_score(y_test, y_pred), "cross_entropy": ce,
        "fit_time_s": fit_time, "predict_time_s": predict_time,
        "total_time_s": fit_time + predict_time,
    }


# =============================================================================
# 4. Optuna Search Spaces & Builders
# =============================================================================
from tabicl import TabICLClassifier

# --- TabICL ---
TABICL_NORM_OPTIONS = ["quantile", "robust", "power", "none"]

def tabicl_search_space(trial: optuna.Trial) -> dict:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 4, 32, step=4),
        "softmax_temperature": trial.suggest_float("softmax_temperature", 0.3, 1.5, step=0.1),
        "outlier_threshold": trial.suggest_float("outlier_threshold", 2.0, 6.0, step=0.5),
        "average_logits": trial.suggest_categorical("average_logits", [True, False]),
    }
    n_norms = trial.suggest_int("n_norm_methods", 1, 2)
    if n_norms == 1:
        params["norm_methods"] = trial.suggest_categorical("norm_method_single", TABICL_NORM_OPTIONS)
    else:
        params["norm_methods"] = [
            trial.suggest_categorical("norm_method_1", TABICL_NORM_OPTIONS),
            trial.suggest_categorical("norm_method_2", TABICL_NORM_OPTIONS)
        ]
    return params


def build_tabicl(params: dict, seed: int = SEED):
    import os
    os.makedirs("./cache/tabicl_offload", exist_ok=True)
    return TabICLClassifier(
        random_state=seed, 
        device=DEVICE, 
        verbose=False, 
        batch_size=4,
        offload_mode="disk",
        disk_offload_dir="./cache/tabicl_offload",
        kv_cache="repr",
        **params
    )

# --- LightGBM ---
def lgbm_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

def build_lgbm(params: dict, seed: int = SEED):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=seed, verbose=-1, **params)

# --- XGBoost ---
def xgb_search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

def build_xgb(params: dict, seed: int = SEED):
    from xgboost import XGBClassifier
    device_arg = {"device": "cuda", "tree_method": "hist"} if DEVICE == "cuda" else {}
    return XGBClassifier(random_state=seed, eval_metric="logloss", verbosity=0, **device_arg, **params)

# --- CatBoost ---
def catboost_search_space(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }

def build_catboost(params: dict, seed: int = SEED):
    from catboost import CatBoostClassifier
    task_type = "GPU" if DEVICE == "cuda" else "CPU"
    return CatBoostClassifier(random_state=seed, verbose=0, task_type=task_type, **params)

# Registro
MODEL_REGISTRY = {
    "lightgbm": {"search_space": lgbm_search_space, "builder": build_lgbm, "n_trials": N_TRIALS_BASELINES},
    "xgboost": {"search_space": xgb_search_space, "builder": build_xgb, "n_trials": N_TRIALS_BASELINES},
    "catboost": {"search_space": catboost_search_space, "builder": build_catboost, "n_trials": N_TRIALS_BASELINES},
    "group_model": {"search_space": tabicl_search_space, "builder": build_tabicl, "n_trials": N_TRIALS_TABICL},
}


# =============================================================================
# 5. Optuna Tuning Engine
# =============================================================================
def tune_model(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray) -> tuple[dict, float]:
    config = MODEL_REGISTRY[model_name]
    all_classes = np.unique(y_train)
    
    # Speedup: Usar split interno (holdout) em vez de 5-fold CV
    # Garantir que o stratify não falhe em classes com 1 amostra
    min_class_count = min(np.bincount(y_train))
    use_stratify = y_train if min_class_count >= 2 else None
    X_tr_inner, X_val_inner, y_tr_inner, y_val_inner = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=use_stratify
    )

    def objective(trial: optuna.Trial) -> float:
        params = config["search_space"](trial)
        estimator = config["builder"](params)
        try:
            estimator.fit(X_tr_inner, y_tr_inner)
            
            if hasattr(estimator, "predict_proba"):
                y_proba = estimator.predict_proba(X_val_inner)
                # Alinhar colunas de probabilidade com as classes do treino
                if hasattr(estimator, "classes_"):
                    pred_classes = estimator.classes_
                else:
                    pred_classes = all_classes
                # Garantir que as classes do val set estejam cobertas
                val_classes = np.unique(y_val_inner)
                if len(val_classes) == 2 and len(pred_classes) == 2:
                    score = float(roc_auc_score(y_val_inner, y_proba[:, 1]))
                elif len(val_classes) < 2:
                    # Apenas uma classe no val set — fallback para accuracy
                    y_pred = estimator.predict(X_val_inner)
                    score = float(accuracy_score(y_val_inner, y_pred))
                else:
                    score = float(roc_auc_score(y_val_inner, y_proba, multi_class="ovo", labels=pred_classes))
            else:
                y_pred = estimator.predict(X_val_inner)
                score = float(accuracy_score(y_val_inner, y_pred))
            return score
        except Exception as e:
            print(f"      [TRIAL WARN] {model_name} trial failed: {e}")
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config["n_trials"], show_progress_bar=False)
    return study.best_params, float(study.best_value)


# =============================================================================
# 6. Statistical Analysis
# =============================================================================
def run_statistical_analysis(df: pd.DataFrame):
    from autorank import autorank, plot_stats
    from baycomp import SignedRankTest

    pivot_auc = df.pivot(index="dataset", columns="model", values="auc_ovo").dropna()
    print(f"\n[STATS] Usando {len(pivot_auc)} datasets completos para análise estatística.")

    # Friedman + Nemenyi
    result = autorank(pivot_auc, alpha=0.05, verbose=False, order="descending")
    print("\n[STATS] Ranking Médio:")
    print(pd.Series(result.rankdf["meanrank"], name="mean_rank").sort_values())

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_stats(result, ax=ax)
    fig.tight_layout()
    fig.savefig(CD_DIAGRAM_PNG, dpi=150)
    plt.close(fig)

    # Bayesian ROPE
    rows = []
    models = list(pivot_auc.columns)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            test = SignedRankTest(x=pivot_auc[a].values, y=pivot_auc[b].values, rope=0.01)
            p_left, p_rope, p_right = test.probs()
            # baycomp: p_left = P(x < y), p_right = P(x > y)
            # So p_right = P(model_a > model_b) = P(a better)
            rows.append({
                "model_a": a, "model_b": b,
                "p_a_better": float(p_right), "p_equivalent": float(p_rope), "p_a_worse": float(p_left),
            })
    pd.DataFrame(rows).to_csv(BAYESIAN_CSV, index=False)


# =============================================================================
# 7. Main Pipeline
# =============================================================================
def main():
    print("=" * 70)
    print("  PIPELINE KAGGLE — TabICL v2 + Baselines com Optuna Tuning")
    print("=" * 70)

    rows, best_params_rows = [], []
    done_task_ids = set()
    n_models = len(MODEL_REGISTRY)

    if RAW_CSV.exists():
        existing = pd.read_csv(RAW_CSV)
        rows = existing.to_dict("records")
        # Só marcar um dataset como "done" se TODOS os 4 modelos foram avaliados
        task_model_counts = existing.groupby("task_id")["model"].nunique()
        done_task_ids = set(task_model_counts[task_model_counts >= n_models].index)
        print(f"\n[RESUME] {len(done_task_ids)} datasets totalmente completados, pulando...")
        # Remover resultados parciais de datasets incompletos
        partial_ids = set(existing["task_id"].unique()) - done_task_ids
        if partial_ids:
            print(f"[RESUME] Removendo resultados parciais de {len(partial_ids)} datasets: {partial_ids}")
            rows = [r for r in rows if r["task_id"] in done_task_ids]
    if BEST_PARAMS_CSV.exists():
        best_params_rows = pd.read_csv(BEST_PARAMS_CSV).to_dict("records")

    n_total = len(RECOMMENDED_TASK_IDS)

    for i, task_id in enumerate(RECOMMENDED_TASK_IDS, 1):
        if task_id in done_task_ids:
            print(f"\n=== [{i}/{n_total}] task_id={task_id}: JÁ COMPLETADO ===")
            continue

        try:
            ds = load_task(task_id)
            print(f"\n{'=' * 70}\n  [{i}/{n_total}] Dataset: {ds.name} (n={ds.n_samples}, p={ds.n_features}, classes={ds.n_classes})\n{'=' * 70}")
        except Exception as e:
            print(f"\n=== [{i}/{n_total}] ERRO ao carregar task_id={task_id}: {e} ===")
            continue

        le_y = LabelEncoder()
        y_encoded = le_y.fit_transform(ds.y)
        min_class_count = min(np.bincount(y_encoded))
        outer_stratify = y_encoded if min_class_count >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(ds.X, y_encoded, test_size=TEST_SIZE, random_state=SEED, stratify=outer_stratify)
        X_train_clean, X_test_clean = preprocess(X_train, X_test)

        for model_name in MODEL_REGISTRY.keys():
            print(f"\n  --- {model_name} ---")
            try:
                # 1. Tuning
                t_tune_start = time.perf_counter()
                best_params, best_cv_score = tune_model(model_name, X_train_clean, y_train)
                tune_time = time.perf_counter() - t_tune_start
                print(f"    [TUNE] Score CV: {best_cv_score:.4f} ({tune_time:.1f}s) | Params: {best_params}")

                best_params_rows.append({
                    "task_id": task_id, "dataset": ds.name, "model": model_name,
                    "best_cv_score": best_cv_score, "tune_time_s": tune_time, "params": str(best_params),
                })

                # 2. Evaluation
                fixed_trial = optuna.trial.FixedTrial(best_params)
                clean_params = MODEL_REGISTRY[model_name]["search_space"](fixed_trial)
                estimator = MODEL_REGISTRY[model_name]["builder"](clean_params)
                metrics = evaluate(estimator, X_train_clean, y_train, X_test_clean, y_test)
                row = {"task_id": task_id, "dataset": ds.name, "model": model_name}
                row.update(metrics)
                rows.append(row)
                print(f"    [TEST] AUC={metrics['auc_ovo']:.4f}, ACC={metrics['accuracy']:.4f}, time={metrics['total_time_s']:.1f}s")

            except Exception as e:
                print(f"    [ERRO] {model_name} falhou: {e}")
                rows.append({
                    "task_id": task_id, "dataset": ds.name, "model": model_name,
                    "auc_ovo": float("nan"), "accuracy": float("nan"), "g_mean": float("nan"), 
                    "cross_entropy": float("nan"), "fit_time_s": float("nan"), 
                    "predict_time_s": float("nan"), "total_time_s": float("nan"),
                })

        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)
        pd.DataFrame(best_params_rows).to_csv(BEST_PARAMS_CSV, index=False)

    print(f"\n{'=' * 70}\n  EXPERIMENTO COMPLETO!\n{'=' * 70}")
    df = pd.read_csv(RAW_CSV)
    
    try:
        run_statistical_analysis(df)
    except Exception as e:
        print(f"\n[ERRO] Análise estatística falhou: {e}")

if __name__ == "__main__":
    main()
