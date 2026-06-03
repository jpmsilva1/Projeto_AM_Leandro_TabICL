"""
=============================================================================
Célula de Continuação — Processa datasets restantes com OOM fix para TabICL
=============================================================================

Cole este código em uma NOVA CÉLULA do Kaggle, ABAIXO da célula principal.
Ele lê os resultados já salvos e processa apenas os datasets que faltam,
com tratamento robusto de GPU Out-Of-Memory para o TabICL.
"""

# ── Tudo que a célula 1 já importou está disponível. ──
# Se por acaso o kernel reiniciou, reimportamos o essencial:
import importlib.util
import os, subprocess, sys, time, warnings

if importlib.util.find_spec("autogluon") is None or importlib.util.find_spec("autogluon.tabular") is None:
    print("[SETUP] Instalando AutoGluon Tabular...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "autogluon.tabular[all]>=1.4,<2.0",
    ])

import numpy as np, pandas as pd, optuna, openml, torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from tabicl import TabICLClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config (idêntica à célula 1) ──
SEED = 42
TEST_SIZE = 0.30
OUTPUT_DIR = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = OUTPUT_DIR / "raw_tuned.csv"
BEST_PARAMS_CSV = OUTPUT_DIR / "best_params.csv"
BAYESIAN_CSV = OUTPUT_DIR / "bayesian_rope.csv"
CD_DIAGRAM_PNG = OUTPUT_DIR / "cd_diagram.png"
N_TRIALS_TABICL = 15
N_TRIALS_BASELINES = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[CONTINUE] Device: {DEVICE}")

RECOMMENDED_TASK_IDS = [
    37, 359955, 15, 49, 3484,
    2, 11, 41, 3022, 3512,
    3892, 168757, 359956, 359967, 359968,
    45, 233214, 233215, 359935, 360932,
    3688, 3945, 168868, 359979, 360113,
    211986, 233211, 233212, 6, 26,
]


# ── Funções utilitárias (copiadas da célula 1) ──
def load_task(task_id, max_retries=5, wait=15):
    """Carrega dataset com retry automático para erros 504/503 do OpenML."""
    for attempt in range(1, max_retries + 1):
        try:
            task = openml.tasks.get_task(task_id, download_data=True)
            dataset = task.get_dataset()
            X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
            return task_id, dataset.name, X, np.asarray(y)
        except Exception as e:
            msg = str(e)
            if any(code in msg for code in ["504", "503", "502", "Connection", "Timeout", "timed out"]):
                if attempt < max_retries:
                    print(f"    [RETRY {attempt}/{max_retries}] OpenML timeout, aguardando {wait}s...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"OpenML não respondeu após {max_retries} tentativas: {e}")
            else:
                raise

def preprocess(X_train, X_test):
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    X_tr, X_te = X_train.copy(), X_test.copy()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr[cat_cols] = enc.fit_transform(X_tr[cat_cols].astype(str))
        X_te[cat_cols] = enc.transform(X_te[cat_cols].astype(str))
    if X_tr.isna().any().any() or X_te.isna().any().any():
        imp = SimpleImputer(strategy="median")
        X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(imp.transform(X_te), columns=X_te.columns, index=X_te.index)
    return X_tr, X_te

def g_mean_score(y_true, y_pred):
    classes = np.unique(y_true)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in classes if (y_true == c).any()]
    return float(np.exp(np.mean(np.log(np.clip(recalls, 1e-12, 1.0))))) if recalls else 0.0

def _compute_metrics_safe(estimator, X, y, pred_classes):
    """Computa auc_ovo, accuracy, g_mean, cross_entropy para um conjunto."""
    y_pred = estimator.predict(X)
    y_proba = estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None

    mask = np.isin(y, pred_classes)
    if not mask.all():
        y = y[mask]
        y_pred = y_pred[mask]
        if y_proba is not None:
            y_proba = y_proba[mask]

    if len(y) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    val_classes = np.unique(y)
    if y_proba is None:
        auc, ce = float("nan"), float("nan")
    elif len(val_classes) == 2 and len(pred_classes) == 2:
        auc = float(roc_auc_score(y, y_proba[:, 1]))
        ce = float(log_loss(y, y_proba, labels=pred_classes))
    elif len(val_classes) < 2:
        auc, ce = float("nan"), float("nan")
    else:
        auc = float(roc_auc_score(y, y_proba, multi_class="ovo", labels=pred_classes))
        ce = float(log_loss(y, y_proba, labels=pred_classes))

    if np.isnan(auc):
        auc = float(accuracy_score(y, y_pred))

    return auc, float(accuracy_score(y, y_pred)), g_mean_score(y, y_pred), ce


def evaluate(estimator, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    pred_classes = estimator.classes_ if hasattr(estimator, "classes_") else np.unique(y_train)

    t0 = time.perf_counter()
    auc, acc, gm, ce = _compute_metrics_safe(estimator, X_test, y_test, pred_classes)
    predict_time = time.perf_counter() - t0

    tr_auc, tr_acc, tr_gm, tr_ce = _compute_metrics_safe(estimator, X_train, y_train, pred_classes)

    return {
        "auc_ovo": auc, "accuracy": acc, "g_mean": gm, "cross_entropy": ce,
        "fit_time_s": fit_time, "predict_time_s": predict_time,
        "total_time_s": fit_time + predict_time,
        "train_auc_ovo": tr_auc, "train_accuracy": tr_acc,
        "train_g_mean": tr_gm, "train_cross_entropy": tr_ce,
    }


# =============================================================================
# TabICL com fallback GPU → CPU e feature selection automática para datasets
# de alta dimensionalidade (> MAX_FEATURES). Apenas o TabICL recebe as features
# reduzidas; todos os outros modelos continuam com o conjunto completo.
# =============================================================================
class TabICLSafe:
    """TabICL com OOM fallback (GPU→CPU) e feature selection automática."""

    MAX_FEATURES = 256  # limiar seguro para GPU T4 do Kaggle

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._device = kwargs.get("device", "cpu")
        self._model = None
        self.classes_ = None
        self._feat_idx = None  # índices das features selecionadas (None = sem redução)

    # ── Feature selection ────────────────────────────────────────────────────
    def _select_fit(self, X):
        """Se n_features > MAX_FEATURES, seleciona top-k por variância."""
        n_feat = X.shape[1]
        if n_feat <= self.MAX_FEATURES:
            self._feat_idx = None
            return X
        variances = np.asarray(X).var(axis=0)
        self._feat_idx = np.argsort(variances)[-self.MAX_FEATURES:]
        print(f"      [FEAT] {n_feat} features → selecionadas top-{self.MAX_FEATURES} por variância")
        return self._apply_selection(X)

    def _apply_selection(self, X):
        if self._feat_idx is None:
            return X
        if hasattr(X, "iloc"):
            return X.iloc[:, self._feat_idx]
        return X[:, self._feat_idx]

    # ── OOM detection ────────────────────────────────────────────────────────
    @staticmethod
    def _is_oom(e):
        if "OutOfMemoryError" in type(e).__name__:
            return True
        msg = str(e).lower()
        return any(k in msg for k in ["out of memory", "cuda error", "memory allocation failed"])

    def _make(self, device):
        kw = {**self._kwargs, "device": device}
        return TabICLClassifier(**kw)

    # ── Fit com fallback GPU → CPU ────────────────────────────────────────────
    def _fit_with_fallback(self, X, y):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self._device != "cpu":
            try:
                self._model = self._make(self._device)
                self._model.fit(X, y)
                return
            except Exception as e:
                if self._is_oom(e):
                    print(f"      [OOM] GPU sem memória. Tentando CPU...")
                    torch.cuda.empty_cache()
                else:
                    raise

        self._model = self._make("cpu")
        self._model.fit(X, y)

    def fit(self, X, y):
        X_sel = self._select_fit(X)
        try:
            self._fit_with_fallback(X_sel, y)
        except Exception as e:
            if self._is_oom(e) and self._feat_idx is None:
                # OOM mesmo sem feature selection — tenta reduzir features agora
                print(f"      [OOM] Reduzindo features para {self.MAX_FEATURES} e retentando...")
                variances = np.asarray(X).var(axis=0)
                self._feat_idx = np.argsort(variances)[-self.MAX_FEATURES:]
                X_sel = self._apply_selection(X)
                self._fit_with_fallback(X_sel, y)
            else:
                raise
        if hasattr(self._model, "classes_"):
            self.classes_ = self._model.classes_
        return self

    def predict(self, X):
        return self._model.predict(self._apply_selection(X))

    def predict_proba(self, X):
        return self._model.predict_proba(self._apply_selection(X))


def build_tabicl_safe(params, seed=SEED):
    import os
    os.makedirs("./cache/tabicl_offload", exist_ok=True)
    return TabICLSafe(
        random_state=seed,
        device=DEVICE,
        verbose=False,
        batch_size=4,
        offload_mode="disk",
        disk_offload_dir="./cache/tabicl_offload",
        kv_cache="repr",
        **params
    )


# =============================================================================
# AutoGluon — wrapper sklearn-compatível
# AutoGluon faz seu próprio tuning interno; não usa Optuna (skip_tune=True).
# time_limit por dataset: default=120s, extreme=300s (proxy para 4h do professor).
# =============================================================================
class AutoGluonWrapper:
    def __init__(self, preset: str, time_limit: int, path_prefix: str):
        self.preset = preset
        self.time_limit = time_limit
        self.path_prefix = path_prefix
        self.predictor_ = None
        self.classes_ = None
        self._label = "__ag_target__"
        self._path = None

    def fit(self, X, y):
        import shutil
        from autogluon.tabular import TabularPredictor
        # Usa id(self) para path único; apaga versão anterior se existir
        self._path = f"/kaggle/working/{self.path_prefix}_{id(self)}"
        if os.path.exists(self._path):
            shutil.rmtree(self._path)
        df = pd.DataFrame(X).copy()
        df[self._label] = y
        self.predictor_ = TabularPredictor(
            label=self._label,
            eval_metric="roc_auc_ovo_macro",
            verbosity=0,
            path=self._path,
        ).fit(df, presets=self.preset, time_limit=self.time_limit)
        self.classes_ = np.array(sorted(np.unique(y)))
        return self

    def predict(self, X):
        return self.predictor_.predict(pd.DataFrame(X)).to_numpy()

    def predict_proba(self, X):
        proba = self.predictor_.predict_proba(pd.DataFrame(X))
        return proba[sorted(proba.columns)].to_numpy()

    def cleanup(self):
        """Remove model files do disco para liberar espaço no Kaggle."""
        import shutil
        if self._path and os.path.exists(self._path):
            shutil.rmtree(self._path, ignore_errors=True)
            self._path = None


def ag_search_space(trial):
    return {}

def build_autogluon_default(params, seed=SEED):
    return AutoGluonWrapper("best_quality", time_limit=120, path_prefix="ag_default")

def build_autogluon_extreme(params, seed=SEED):
    # Proxy do preset extreme (4h do professor); limitado a 300s por dataset no Kaggle.
    return AutoGluonWrapper("extreme_quality", time_limit=300, path_prefix="ag_extreme")


# =============================================================================
# Search Spaces (idênticos à célula 1)
# =============================================================================
TABICL_NORM_OPTIONS = ["quantile", "robust", "power", "none"]

def tabicl_search_space(trial):
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
            trial.suggest_categorical("norm_method_2", TABICL_NORM_OPTIONS),
        ]
    return params

def lgbm_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }

def build_lgbm(params, seed=SEED):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(random_state=seed, verbose=-1, **params)

def xgb_search_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }

def build_xgb(params, seed=SEED):
    from xgboost import XGBClassifier
    dev = {"device": "cuda", "tree_method": "hist"} if DEVICE == "cuda" else {}
    return XGBClassifier(random_state=seed, eval_metric="logloss", verbosity=0, **dev, **params)

def catboost_search_space(trial):
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
    }

def build_catboost(params, seed=SEED):
    from catboost import CatBoostClassifier
    tt = "GPU" if DEVICE == "cuda" else "CPU"
    return CatBoostClassifier(random_state=seed, verbose=0, task_type=tt, **params)


MODEL_REGISTRY = {
    "lightgbm":          {"search_space": lgbm_search_space,    "builder": build_lgbm,              "n_trials": N_TRIALS_BASELINES, "skip_tune": False},
    "xgboost":           {"search_space": xgb_search_space,     "builder": build_xgb,               "n_trials": N_TRIALS_BASELINES, "skip_tune": False},
    "catboost":          {"search_space": catboost_search_space, "builder": build_catboost,          "n_trials": N_TRIALS_BASELINES, "skip_tune": False},
    "group_model":       {"search_space": tabicl_search_space,  "builder": build_tabicl_safe,       "n_trials": N_TRIALS_TABICL,    "skip_tune": False},
    "autogluon_default": {"search_space": ag_search_space,      "builder": build_autogluon_default, "n_trials": 1,                  "skip_tune": True},
    "autogluon_extreme": {"search_space": ag_search_space,      "builder": build_autogluon_extreme, "n_trials": 1,                  "skip_tune": True},
}


# =============================================================================
# Tuning (holdout, com OOM safety)
# =============================================================================
def tune_model(model_name, X_train, y_train):
    config = MODEL_REGISTRY[model_name]
    all_classes = np.unique(y_train)

    min_cc = min(np.bincount(y_train))
    strat = y_train if min_cc >= 2 else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=strat
    )

    def objective(trial):
        params = config["search_space"](trial)
        est = config["builder"](params)
        try:
            est.fit(X_tr, y_tr)
            
            # Prevenir crash em datasets com muitas classes raras
            pred_classes = est.classes_ if hasattr(est, "classes_") else all_classes
            y_v = y_val
            X_v = X_val
            mask = np.isin(y_v, pred_classes)
            if not mask.all():
                y_v = y_v[mask]
                X_v = X_v[mask]
                
            if len(y_v) == 0:
                return 0.0
                
            if hasattr(est, "predict_proba"):
                y_proba = est.predict_proba(X_v)
                val_classes = np.unique(y_v)
                if len(val_classes) == 2 and len(pred_classes) == 2:
                    return float(roc_auc_score(y_v, y_proba[:, 1]))
                elif len(val_classes) < 2:
                    return float(accuracy_score(y_v, est.predict(X_v)))
                else:
                    return float(roc_auc_score(y_v, y_proba, multi_class="ovo", labels=pred_classes))
            else:
                return float(accuracy_score(y_v, est.predict(X_v)))
        except Exception as e:
            print(f"      [TRIAL WARN] {model_name}: {e}")
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=config["n_trials"], show_progress_bar=False)
    return study.best_params, float(study.best_value)


# =============================================================================
# Statistical Analysis
# =============================================================================
def run_statistical_analysis(df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from autorank import autorank, plot_stats
    from baycomp import SignedRankTest

    pivot = df.pivot(index="dataset", columns="model", values="auc_ovo").dropna()
    print(f"\n[STATS] Usando {len(pivot)} datasets completos para análise estatística.")

    result = autorank(pivot, alpha=0.05, verbose=False, order="descending")
    print("\n[STATS] Ranking Médio:")
    print(pd.Series(result.rankdf["meanrank"], name="mean_rank").sort_values())

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_stats(result, ax=ax)
    fig.tight_layout()
    fig.savefig(CD_DIAGRAM_PNG, dpi=150)
    plt.close(fig)

    rows = []
    models = list(pivot.columns)
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            test = SignedRankTest(x=pivot[a].values, y=pivot[b].values, rope=0.01)
            p_left, p_rope, p_right = test.probs()
            rows.append({
                "model_a": a, "model_b": b,
                "p_a_better": float(p_right), "p_equivalent": float(p_rope), "p_a_worse": float(p_left),
            })
    pd.DataFrame(rows).to_csv(BAYESIAN_CSV, index=False)


# =============================================================================
# MAIN — Continuation
# =============================================================================
print("=" * 70)
print("  CONTINUAÇÃO — Processando datasets restantes com OOM fix")
print("=" * 70)

rows, best_params_rows = [], []
done_task_ids = set()
n_models = len(MODEL_REGISTRY)

if RAW_CSV.exists():
    existing = pd.read_csv(RAW_CSV)
    rows = existing.to_dict("records")
    # Só considerar "done" se TODOS os modelos produziram AUC válida.
    valid = existing.dropna(subset=["auc_ovo"])
    counts = valid.groupby("task_id")["model"].nunique()
    done_task_ids = set(counts[counts >= n_models].index)
    # Limpar resultados parciais
    partial = set(existing["task_id"].unique()) - done_task_ids
    if partial:
        print(f"[RESUME] Removendo resultados parciais de {len(partial)} dataset(s): {partial}")
        rows = [r for r in rows if r["task_id"] in done_task_ids]
    print(f"[RESUME] {len(done_task_ids)} datasets já completos. Faltam: {len(RECOMMENDED_TASK_IDS) - len(done_task_ids)}")

if BEST_PARAMS_CSV.exists():
    best_params_rows = pd.read_csv(BEST_PARAMS_CSV).to_dict("records")
    best_params_rows = [r for r in best_params_rows if r["task_id"] in done_task_ids]

n_total = len(RECOMMENDED_TASK_IDS)

for i, task_id in enumerate(RECOMMENDED_TASK_IDS, 1):
    if task_id in done_task_ids:
        continue

    try:
        tid, name, X, y = load_task(task_id)
        n_s  = X.shape[0]
        n_f  = X.shape[1]
        n_c  = len(np.unique(y))
        n_cat = int(X.select_dtypes(exclude=[np.number]).shape[1])
        has_missing = bool(X.isna().any().any())
        print(f"\n{'=' * 70}\n  [{i}/{n_total}] {name} (n={n_s}, p={n_f}, classes={n_c}, "
              f"cat={n_cat}, missing={has_missing})\n{'=' * 70}")
    except Exception as e:
        print(f"\n=== [{i}/{n_total}] ERRO ao carregar task_id={task_id}: {e} ===")
        continue

    le_y = LabelEncoder()
    y_enc = le_y.fit_transform(y)
    min_cc = min(np.bincount(y_enc))
    strat = y_enc if min_cc >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=TEST_SIZE, random_state=SEED, stratify=strat)
    X_train_c, X_test_c = preprocess(X_train, X_test)

    for model_name in MODEL_REGISTRY:
        print(f"\n  --- {model_name} ---")
        est = None
        try:
            # Tuning: AutoGluon faz tuning interno — pula o Optuna
            if MODEL_REGISTRY[model_name]["skip_tune"]:
                best_params, best_score, tune_time = {}, float("nan"), 0.0
                print(f"    [TUNE] Skipped (AutoGluon faz tuning interno)")
            else:
                t0 = time.perf_counter()
                best_params, best_score = tune_model(model_name, X_train_c, y_train)
                tune_time = time.perf_counter() - t0
                print(f"    [TUNE] Score: {best_score:.4f} ({tune_time:.1f}s)")

            best_params_rows.append({
                "task_id": task_id, "dataset": name, "model": model_name,
                "best_cv_score": best_score, "tune_time_s": tune_time, "params": str(best_params),
            })

            fixed = optuna.trial.FixedTrial(best_params)
            clean = MODEL_REGISTRY[model_name]["search_space"](fixed)
            est = MODEL_REGISTRY[model_name]["builder"](clean)
            metrics = evaluate(est, X_train_c, y_train, X_test_c, y_test)

            row = {
                "task_id": task_id, "dataset": name, "model": model_name,
                "n_samples": n_s, "n_features": n_f, "n_classes": n_c,
                "n_categorical": n_cat, "has_missing": has_missing,
            }
            row.update(metrics)
            row["tune_time_s"] = tune_time
            row["total_pipeline_time_s"] = tune_time + metrics["total_time_s"]
            rows.append(row)

            print(f"    [TRAIN] AUC={metrics['train_auc_ovo']:.4f}, ACC={metrics['train_accuracy']:.4f}")
            print(f"    [TEST]  AUC={metrics['auc_ovo']:.4f}, ACC={metrics['accuracy']:.4f}, "
                  f"time={row['total_pipeline_time_s']:.1f}s (tune+fit+pred)")

        except Exception as e:
            print(f"    [ERRO] {model_name} falhou: {e}")
            rows.append({
                "task_id": task_id, "dataset": name, "model": model_name,
                "n_samples": n_s, "n_features": n_f, "n_classes": n_c,
                "n_categorical": n_cat, "has_missing": has_missing,
                "auc_ovo": float("nan"), "accuracy": float("nan"), "g_mean": float("nan"),
                "cross_entropy": float("nan"), "fit_time_s": float("nan"),
                "predict_time_s": float("nan"), "total_time_s": float("nan"),
                "train_auc_ovo": float("nan"), "train_accuracy": float("nan"),
                "train_g_mean": float("nan"), "train_cross_entropy": float("nan"),
                "tune_time_s": float("nan"), "total_pipeline_time_s": float("nan"),
            })
        finally:
            # Liberar disco mesmo quando o AutoGluon falhar durante o fit.
            if est is not None and hasattr(est, "cleanup"):
                est.cleanup()

    # Salvar após cada dataset
    pd.DataFrame(rows).to_csv(RAW_CSV, index=False)
    pd.DataFrame(best_params_rows).to_csv(BEST_PARAMS_CSV, index=False)

print(f"\n{'=' * 70}\n  EXPERIMENTO COMPLETO!\n{'=' * 70}")
df = pd.read_csv(RAW_CSV)

try:
    run_statistical_analysis(df)
    print("[STATS] Análise estatística salva com sucesso!")
except Exception as e:
    print(f"[ERRO] Análise estatística falhou: {e}")
