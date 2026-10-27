"""WESAD subject discovery, loading, and structural inspection helpers."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

LABEL_NAMES = {
    0: "transition_or_undefined",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

# Rates documented by the WESAD UCI record and original paper.
SAMPLE_RATES_HZ = {
    "chest": {"default": 700.0},
    "wrist": {"BVP": 64.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0},
}


def find_subject_files(data_root: str | Path) -> list[Path]:
    """Find official-style S<n>/S<n>.pkl files directly under data_root."""
    root = Path(data_root).expanduser()
    if not root.is_dir():
        return []
    files = []
    for folder in root.iterdir():
        if folder.is_dir() and re.fullmatch(r"S\d+", folder.name):
            candidate = folder / f"{folder.name}.pkl"
            if candidate.is_file():
                files.append(candidate)
    return sorted(files, key=lambda p: int(p.parent.name[1:]))


def load_subject(path: str | Path) -> dict[str, Any]:
    """Load one trusted official WESAD pickle and validate its top-level format.

    Pickle can execute code while loading. Only use files from the official
    dataset archive obtained from a trusted source; never load arbitrary pickle
    files supplied by an unknown person or website.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"WESAD subject pickle not found: {path}")
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary in {path.name}; found {type(payload).__name__}.")
    missing = {"signal", "label"} - set(payload)
    if missing:
        raise ValueError(f"{path.name} is missing expected top-level keys: {sorted(missing)}")
    if not isinstance(payload["signal"], dict):
        raise ValueError(f"{path.name}['signal'] is not a dictionary of devices/channels.")
    return payload


def _channel_rate(device: str, channel: str) -> float | None:
    device_rates = SAMPLE_RATES_HZ.get(device.lower(), {})
    if "default" in device_rates:
        return device_rates["default"]
    for key, rate in device_rates.items():
        if key.casefold() == channel.casefold():
            return rate
    return None


def inspect_subject(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the loaded payload plus one metadata row per device channel."""
    path = Path(path)
    payload = load_subject(path)
    labels = np.asarray(payload["label"]).reshape(-1)
    rows: list[dict[str, Any]] = []
    subject_id = str(payload.get("subject", path.parent.name))
    for device, channels in payload["signal"].items():
        if not isinstance(channels, dict):
            rows.append({"subject": subject_id, "device": str(device), "channel": None,
                         "shape": repr(np.asarray(channels).shape), "samples": None,
                         "sample_rate_hz": None, "missing_values": None,
                         "nonfinite_values": None, "matches_label_length": False})
            continue
        for channel, raw_values in channels.items():
            values = np.asarray(raw_values)
            numeric = np.issubdtype(values.dtype, np.number)
            nonfinite = int((~np.isfinite(values)).sum()) if numeric else None
            if values.ndim == 0:
                sample_count = 1
            else:
                sample_count = int(values.shape[0])
            rows.append({
                "subject": subject_id,
                "device": str(device),
                "channel": str(channel),
                "shape": str(values.shape),
                "samples": sample_count,
                "sample_rate_hz": _channel_rate(str(device), str(channel)),
                "missing_values": nonfinite,
                "nonfinite_values": nonfinite,
                "matches_label_length": sample_count == len(labels),
            })
    return payload, rows


def label_name(label: int) -> str:
    """Return a known label name or an explicit unknown-code description."""
    return LABEL_NAMES.get(int(label), f"other_or_unmapped_{int(label)}")
