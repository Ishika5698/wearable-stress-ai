"""Participant-independent evaluation for a majority baseline and random forest."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

LABEL_ORDER = (1, 2, 3)
CLASS_NAMES = ("Baseline", "Stress", "Amusement")


def evaluate_grouped_cv(
    table: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return out-of-fold metrics, predictions, and split audit for grouped CV."""
    required = {"subject", "label"}
    if missing := required - set(table.columns):
        raise ValueError(f"Feature table is missing columns: {sorted(missing)}")
    feature_columns = [
        column for column in table.select_dtypes(include=[np.number]).columns
        if column != "label" and column != "window_start_seconds"
    ]
    if not feature_columns:
        raise ValueError("No numeric signal features were found.")
    X = table[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = table["label"].to_numpy(dtype=int)
    groups = table["subject"].astype(str).to_numpy()
    classes = set(np.unique(y))
    if classes != set(LABEL_ORDER):
        raise ValueError(
            f"Expected target labels {LABEL_ORDER} (baseline/stress/amusement); got {sorted(classes)}. "
            "Check the accompanying WESAD README and label mapping."
        )
    unique_groups = np.unique(groups)
    groups_per_class = {
        label: np.unique(groups[y == label]).size for label in LABEL_ORDER
    }
    split_count = min(n_splits, unique_groups.size, min(groups_per_class.values()))
    if split_count < 2:
        raise ValueError(
            "At least two distinct participants per class are required for grouped validation."
        )
    cv = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=random_state)
    splits = list(cv.split(X, y, groups))
    split_audit = []
    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise AssertionError(f"Participant leakage in fold {fold}: {sorted(overlap)}")
        if set(np.unique(y[train_idx])) != set(LABEL_ORDER):
            raise ValueError(f"Fold {fold} training data is missing at least one target class.")
        split_audit.append({
            "fold": fold,
            "train_subjects": ",".join(sorted(set(groups[train_idx]))),
            "test_subjects": ",".join(sorted(set(groups[test_idx]))),
            "subject_overlap": len(overlap),
        })

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Random forest": make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample",
                min_samples_leaf=2, random_state=random_state, n_jobs=-1,
            ),
        ),
    }
    metrics_rows, per_class_rows, confusion_rows, fold_rows = [], [], [], []
    predictions_by_model = {}
    for model_name, estimator in models.items():
        oof = np.full(y.shape, fill_value=-1, dtype=int)
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            fitted = clone(estimator).fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = fitted.predict(X.iloc[test_idx])
            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "accuracy": accuracy_score(y[test_idx], oof[test_idx]),
                "macro_f1": f1_score(y[test_idx], oof[test_idx], labels=LABEL_ORDER,
                                      average="macro", zero_division=0),
                "test_subjects": len(np.unique(groups[test_idx])),
            })
        if np.any(oof < 0):
            raise AssertionError("Some windows did not receive an out-of-fold prediction.")
        predictions_by_model[model_name] = oof
        metrics_rows.append({
            "model": model_name,
            "accuracy": accuracy_score(y, oof),
            "macro_f1": f1_score(y, oof, labels=LABEL_ORDER, average="macro", zero_division=0),
        })
        precision, recall, f1, support = precision_recall_fscore_support(
            y, oof, labels=LABEL_ORDER, zero_division=0
        )
        for index, label in enumerate(LABEL_ORDER):
            per_class_rows.append({
                "model": model_name,
                "label": label,
                "class_name": CLASS_NAMES[index],
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
                "support": support[index],
            })
        matrix = confusion_matrix(y, oof, labels=LABEL_ORDER)
        for actual_index, actual_label in enumerate(LABEL_ORDER):
            for predicted_index, predicted_label in enumerate(LABEL_ORDER):
                confusion_rows.append({
                    "model": model_name,
                    "actual": CLASS_NAMES[actual_index],
                    "predicted": CLASS_NAMES[predicted_index],
                    "count": int(matrix[actual_index, predicted_index]),
                })
    return {
        "metrics": pd.DataFrame(metrics_rows),
        "per_class": pd.DataFrame(per_class_rows),
        "confusion": pd.DataFrame(confusion_rows),
        "fold_metrics": pd.DataFrame(fold_rows),
        "split_audit": pd.DataFrame(split_audit),
        "predictions": predictions_by_model,
        "y_true": y,
        "groups": groups,
        "feature_columns": feature_columns,
        "label_order": LABEL_ORDER,
        "class_names": CLASS_NAMES,
    }


def save_evaluation(result: dict[str, Any], report_dir: str) -> None:
    """Save measured results only after evaluation has run on real data."""
    from pathlib import Path

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(out / "model_metrics.csv", index=False)
    result["per_class"].to_csv(out / "per_class_metrics.csv", index=False)
    result["fold_metrics"].to_csv(out / "fold_metrics.csv", index=False)
    result["confusion"].to_csv(out / "confusion_matrices.csv", index=False)
