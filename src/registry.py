"""Keeps track of model versions in models/registry.json (which one is live, metrics, threshold)."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import config as C


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_registry(path: Path = C.REGISTRY_PATH) -> dict:
    if not path.exists():
        return {"current": None, "versions": {}}
    with open(path) as fh:
        return json.load(fh)


def save_registry(registry: dict, path: Path = C.REGISTRY_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # write to a temp file then rename, so other containers never read a half-written file
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".registry-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(registry, fh, indent=2)
    os.chmod(tmp, 0o644)  # mkstemp makes it 0600 by default
    os.replace(tmp, path)


def current_entry(registry: dict | None = None) -> dict:
    registry = registry or load_registry()
    if not registry.get("current"):
        raise FileNotFoundError(f"no model registered in {C.REGISTRY_PATH}; run the notebook first")
    return registry["versions"][registry["current"]]


def model_path(entry: dict) -> Path:
    return C.MODELS_DIR / entry["path"]


def next_version(registry: dict) -> str:
    numbers = [int(v.lstrip("v")) for v in registry["versions"] if v.lstrip("v").isdigit()]
    return f"v{max(numbers, default=0) + 1}"


def register(*, filename: str, test_metrics: dict, threshold: float, trigger: str,
             parent: str | None = None, history: dict | None = None, train_samples: int | None = None,
             prevalence: float | None = None, notes: str = "", make_current: bool = True,
             version: str | None = None) -> str:
    registry = load_registry()
    version = version or next_version(registry)
    registry["versions"][version] = {
        "version": version,
        "path": filename,
        "created_at": _now(),
        "trigger": trigger,
        "parent": parent,
        "threshold": float(threshold),
        "test_metrics": test_metrics,
        "history": history or {},
        "train_samples": train_samples,
        "prevalence": prevalence,  # share of abnormal clips, used when re-tuning the threshold
        "notes": notes,
        "preprocessing": {
            "sample_rate": C.SAMPLE_RATE, "clip_seconds": C.CLIP_SECONDS, "n_fft": C.N_FFT,
            "hop_length": C.HOP_LENGTH, "n_mels": C.N_MELS, "classes": list(C.CLASSES),
        },
    }
    if make_current:
        registry["current"] = version
    save_registry(registry)
    return version


def activate(version: str) -> dict:
    registry = load_registry()
    if version not in registry["versions"]:
        raise KeyError(version)
    if not model_path(registry["versions"][version]).exists():
        raise FileNotFoundError(f"model file for {version} is missing")
    registry["current"] = version
    save_registry(registry)
    return registry["versions"][version]
