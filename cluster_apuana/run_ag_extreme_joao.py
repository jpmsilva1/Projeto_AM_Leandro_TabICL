"""
===================================================================
TATICA ENXAME — JOAO (jpms5)
Batch: 10 datasets LARGE (os mais pesados)
AutoGluon Extreme com 4 HORAS de limite por dataset
Roda SEM GPU (apenas CPU) para nao conflitar com o Job V2
===================================================================
"""
import os
import sys
import time
import signal
import warnings
import logging
import multiprocessing
import numpy as np
import pandas as pd
import random
import openml

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# --- CONFIGURACOES ---
SEED = 42
AG_EXTREME_TIME_LIMIT = 14400  # 4 HORAS (em segundos)
ALARM_LIMIT = AG_EXTREME_TIME_LIMIT + 3600  # 5h margem de seguranca

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("ag_extreme_joao.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

num_cpus_env = os.environ.get("SLURM_CPUS_PER_TASK")
NUM_CPUS = int(num_cpus_env) if num_cpus_env else multiprocessing.cpu_count()
logging.info(f"CPUs disponiveis: {NUM_CPUS}")

os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPUS)

RESULTS_FILE = "ag_extreme_joao.csv"

# ========================================================
# BATCH JOAO — 10 datasets LARGE
# ========================================================
DATASETS = [
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


def timeout_handler(signum, frame):
    raise TimeoutError("Tempo limite de seguranca atingido!")


def build_preprocessor(cat_indicator, X_columns):
    cat_cols = [col for i, col in enumerate(X_columns) if cat_indicator[i]]
    num_cols = [col for i, col in enumerate(X_columns) if not cat_indicator[i]]
    transformers = []
    if cat_cols:
        transformers.append(("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols))
    if num_cols:
        transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


def compute_auc(y_true, y_proba, n_classes):
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_proba, multi_class="ovo", labels=np.arange(n_classes)))
    except Exception as e:
        logging.warning(f"AUC failed: {e}")
        return np.nan


def g_mean_score(y_true, y_pred):
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


def compute_cross_entropy(y_true, y_proba, n_classes):
    try:
        return float(log_loss(y_true, y_proba, labels=np.arange(n_classes)))
    except:
        return np.nan


def run_autogluon_extreme(X_train, y_train, X_test, y_test, n_classes):
    from autogluon.tabular import TabularPredictor
    import tempfile, shutil

    logging.info(f"  ⚙️  Iniciando AutoGluon Extreme (limite=4h / {AG_EXTREME_TIME_LIMIT}s)...")

    ag_path = tempfile.mkdtemp()
    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test)

    t0 = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(ALARM_LIMIT)

        ag_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_macro"
        ag_problem_type = "binary" if n_classes == 2 else "multiclass"

        predictor = TabularPredictor(
            label="target", eval_metric=ag_metric,
            problem_type=ag_problem_type, path=ag_path, verbosity=0
        )
        predictor.fit(
            train_data=train_df, presets="best_quality",
            time_limit=AG_EXTREME_TIME_LIMIT,
            num_gpus=0,
            num_cpus=NUM_CPUS,
            ag_args_fit={"num_cpus": NUM_CPUS}
        )

        signal.alarm(0)

        y_pred = predictor.predict(test_df).values
        y_proba = predictor.predict_proba(test_df).values

        total_time = time.perf_counter() - t0
        acc = accuracy_score(y_test, y_pred)
        gmean = g_mean_score(y_test, y_pred)
        auc = compute_auc(y_test, y_proba, n_classes)
        ce = compute_cross_entropy(y_test, y_proba, n_classes)

        logging.info(f"  ✅ AutoGluon_Extreme_4h | ACC={acc:.4f} | AUC={auc:.4f} | Tempo={total_time:.1f}s")
        shutil.rmtree(ag_path, ignore_errors=True)

        return {
            "model": "AutoGluon_Extreme_4h",
            "ACC": round(acc, 6), "AUC_OVO": round(auc, 6),
            "G_Mean": round(gmean, 6), "CE": round(ce, 6),
            "total_time_s": round(total_time, 2)
        }
    except Exception as e:
        signal.alarm(0)
        shutil.rmtree(ag_path, ignore_errors=True)
        raise e


def main():
    logging.info("=" * 65)
    logging.info("🐝 TATICA ENXAME — JOAO (Batch LARGE)")
    logging.info("=" * 65)

    all_results = []
    done_datasets = set()
    if os.path.exists(RESULTS_FILE):
        df_done = pd.read_csv(RESULTS_FILE)
        all_results = df_done.to_dict('records')
        done_datasets = set(df_done['dataset'].unique())

    for ds in DATASETS:
        if ds['name'] in done_datasets:
            logging.info(f"  ⏭️  Pulando {ds['name']} (ja concluido)")
            continue

        logging.info(f"\n{'='*65}")
        logging.info(f"  📁 Dataset: {ds['name']} (tid={ds['tid']}, regime={ds['regime']})")
        logging.info(f"{'='*65}")

        try:
            try:
                task = openml.tasks.get_task(ds["tid"])
                dataset = openml.datasets.get_dataset(
                    task.dataset_id, download_data=True,
                    download_qualities=False, download_features_meta_data=True
                )
            except openml.exceptions.OpenMLServerException as e:
                if "Unknown task" in str(e):
                    logging.warning(f"  Task {ds['tid']} inexistente. Usando Dataset ID direto...")
                    dataset = openml.datasets.get_dataset(
                        ds["tid"], download_data=True,
                        download_qualities=False, download_features_meta_data=True
                    )
                else:
                    raise e

            X, y, cat_indicator, _ = dataset.get_data(target=dataset.default_target_attribute)

            y_clean = y.dropna()
            X = X.loc[y_clean.index]

            le = LabelEncoder()
            y_encoded = le.fit_transform(y_clean)
            n_classes = len(le.classes_)

            try:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=SEED, stratify=y_encoded
                )
            except ValueError:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=0.3, random_state=SEED
                )

            preprocessor = build_preprocessor(cat_indicator, X.columns)
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)

            if hasattr(X_train, "toarray"):
                X_train = X_train.toarray()
                X_test = X_test.toarray()

            X_train = X_train.astype(float)
            X_test = X_test.astype(float)

            res = run_autogluon_extreme(X_train, y_train, X_test, y_test, n_classes)
            res['dataset'] = ds['name']
            res['regime'] = ds['regime']
            all_results.append(res)

            pd.DataFrame(all_results).to_csv(RESULTS_FILE, index=False)
            logging.info(f"  💾 Resultado salvo em {RESULTS_FILE}")

        except Exception as e:
            logging.error(f"  ❌ DATASET {ds['name']} FALHOU: {e}", exc_info=True)

    logging.info("\n" + "=" * 65)
    logging.info("🏁 BATCH JOAO FINALIZADO!")
    logging.info("=" * 65)


if __name__ == "__main__":
    main()
