"""Strict-label windowing and simple time-domain feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import iqr

from .wesad import find_subject_files, load_subject

CHEST_RATE_HZ = 700.0
MODEL_CHANNELS = ("ECG", "EDA", "TEMP")
TARGET_LABELS = (1, 2, 3)  # baseline, stress, amusement


def _case_insensitive_key(mapping: dict, desired: str) -> str:
    matches = [str(key) for key in mapping if str(key).casefold() == desired.casefold()]
    if not matches:
        raise KeyError(f"Channel {desired!r} not found. Available channels: {list(mapping)}")
    return matches[0]


def _one_dimensional_channel(raw: Any, channel: str, subject: str) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim == 2 and 1 in values.shape:
        values = values.reshape(-1)
    if values.ndim != 1:
        raise ValueError(
            f"{subject} chest channel {channel} has shape {values.shape}; "
            "the initial model expects scalar physiological channels."
        )
    return values


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    missing_fraction = 1.0 - (finite.size / values.size) if values.size else 1.0
    names = ("mean", "std", "min", "max", "median", "iqr", "rms")
    result = {f"{prefix}_{name}": float("nan") for name in names}
    result[f"{prefix}_missing_fraction"] = float(missing_fraction)
    # Leave statistics missing when less than 80% of a window is finite. The
    # modeling imputer is fitted within each training fold, preventing leakage.
    if finite.size >= 0.8 * values.size and finite.size:
        result.update({
            f"{prefix}_mean": float(np.mean(finite)),
            f"{prefix}_std": float(np.std(finite)),
            f"{prefix}_min": float(np.min(finite)),
            f"{prefix}_max": float(np.max(finite)),
            f"{prefix}_median": float(np.median(finite)),
            f"{prefix}_iqr": float(iqr(finite)),
            f"{prefix}_rms": float(np.sqrt(np.mean(finite ** 2))),
        })
    return result


def extract_subject_features(
    payload: dict[str, Any],
    subject_id: str,
    window_seconds: float = 30.0,
    step_seconds: float = 15.0,
    channels: tuple[str, ...] = MODEL_CHANNELS,
    target_labels: tuple[int, ...] = TARGET_LABELS,
) -> pd.DataFrame:
    """Create participant-tagged features from label-pure chest windows.

    The label vector is documented on the 700 Hz chest clock. Windows that
    include any transition, excluded condition, or changing class are omitted.
    Overlapping windows remain grouped by subject during validation.
    """
    labels = np.asarray(payload["label"], dtype=int).reshape(-1)
    chest_key = _case_insensitive_key(payload["signal"], "chest")
    chest = payload["signal"][chest_key]
    actual_keys = {channel: _case_insensitive_key(chest, channel) for channel in channels}
    series = {
        channel: _one_dimensional_channel(chest[key], channel, subject_id)
        for channel, key in actual_keys.items()
    }
    for channel, values in series.items():
        if len(values) != len(labels):
            raise ValueError(
                f"{subject_id}: chest {channel} has {len(values)} samples but labels have "
                f"{len(labels)}. Check the released pickle and synchronization metadata."
            )
    window_size = int(round(window_seconds * CHEST_RATE_HZ))
    step_size = int(round(step_seconds * CHEST_RATE_HZ))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window and step durations must be positive.")

    rows = []
    for start in range(0, len(labels) - window_size + 1, step_size):
        label_window = labels[start:start + window_size]
        label = int(label_window[0])
        if label not in target_labels or not np.all(label_window == label):
            continue
        row: dict[str, Any] = {
            "subject": str(subject_id),
            "label": label,
            "window_start_seconds": start / CHEST_RATE_HZ,
        }
        for channel, values in series.items():
            row.update(_summary(values[start:start + window_size], channel.lower()))
        rows.append(row)
    return pd.DataFrame(rows)


def extract_dataset_features(
    data_root: str | Path,
    window_seconds: float = 30.0,
    step_seconds: float = 15.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process one participant pickle at a time and return features + inventory."""
    files = find_subject_files(data_root)
    if not files:
        raise FileNotFoundError(
            f"No subject pickle files found under {Path(data_root)}. "
            "Expected folders such as S2/S2.pkl; see data/README.md."
        )
    feature_tables = []
    for path in files:
        payload = load_subject(path)
        subject_id = str(payload.get("subject", path.parent.name))
        table = extract_subject_features(payload, subject_id, window_seconds, step_seconds)
        if not table.empty:
            feature_tables.append(table)
    if not feature_tables:
        raise ValueError(
            "No label-pure target windows were produced. Confirm label codes and "
            "sampling rates against the README included with your WESAD release."
        )
    return pd.concat(feature_tables, ignore_index=True), pd.DataFrame()
