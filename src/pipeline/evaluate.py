"""Calculo das metricas exigidas: AUC OVO, ACC, G-Mean, Cross-Entropy e tempo."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    auc_ovo: float
    accuracy: float
    g_mean: float
    cross_entropy: float
    fit_time_s: float
    predict_time_s: float

    def to_dict(self) -> dict[str, float]:
        return {
            "auc_ovo": self.auc_ovo,
            "accuracy": self.accuracy,
            "g_mean": self.g_mean,
            "cross_entropy": self.cross_entropy,
            "fit_time_s": self.fit_time_s,
            "predict_time_s": self.predict_time_s,
            "total_time_s": self.fit_time_s + self.predict_time_s,
        }


def g_mean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """G-Mean: media geometrica do recall por classe."""
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


def fit_predict_evaluate(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> EvaluationResult:
    """Treina e avalia um estimador, medindo tempos de fit e predict."""
    t0 = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = estimator.predict(X_test)
    if hasattr(estimator, "predict_proba"):
        y_proba = estimator.predict_proba(X_test)
    else:
        y_proba = None
    predict_time_s = time.perf_counter() - t0

    classes = np.unique(np.concatenate([y_train, y_test]))
    multi_class = "ovo" if classes.size > 2 else "raise"
    if y_proba is None:
        auc = float("nan")
        ce = float("nan")
    else:
        if classes.size == 2:
            auc = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            auc = float(
                roc_auc_score(y_test, y_proba, multi_class=multi_class, labels=classes)
            )
        ce = float(log_loss(y_test, y_proba, labels=classes))

    return EvaluationResult(
        auc_ovo=auc,
        accuracy=float(accuracy_score(y_test, y_pred)),
        g_mean=g_mean_score(y_test, y_pred),
        cross_entropy=ce,
        fit_time_s=fit_time_s,
        predict_time_s=predict_time_s,
    )
