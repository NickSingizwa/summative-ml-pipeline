"""Loads the live model from the registry and predicts on audio files.

Example:
    from src.prediction import predict_file
    predict_file("data/samples/abnormal_id_02_b00000003.wav")
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from . import config as C
from . import registry
from .model import load_model
from .preprocessing import audio_to_model_input


def _configure_threads() -> None:
    # used in docker-compose to pin each container to 1 thread for the load test
    threads = int(os.getenv("TF_NUM_THREADS", "0"))
    if threads > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(threads)
        except RuntimeError:
            pass  # too late, TF already started


_configure_threads()


class Predictor:
    def __init__(self, registry_path: Path = C.REGISTRY_PATH):
        self.registry_path = registry_path
        self._lock = threading.Lock()
        self._mtime: int | None = None
        self.model = None
        self.version: str | None = None
        self.threshold = 0.5
        self.entry: dict = {}
        self.reload_if_changed()

    def reload_if_changed(self) -> bool:
        # checking the file's mtime is cheap, so this runs before every prediction.
        # that way a model promoted by a retrain (even in another container) gets picked up
        mtime = self.registry_path.stat().st_mtime_ns
        if mtime == self._mtime:
            return False
        with self._lock:
            if mtime == self._mtime:
                return False
            reg = registry.load_registry(self.registry_path)
            entry = registry.current_entry(reg)
            changed = entry["version"] != self.version
            if changed:
                model = load_model(registry.model_path(entry))
                model(np.zeros((1, *C.INPUT_SHAPE), np.float32), training=False)  # warm up so the first request isn't slow
                self.model = model
            self.version, self.entry = entry["version"], entry
            self.threshold = float(entry.get("threshold", 0.5))
            self._mtime = mtime
            return changed

    def predict_tensor(self, x: np.ndarray) -> np.ndarray:
        # calling the model directly is a lot faster than model.predict() for small batches
        model = self.model
        return np.asarray(model(np.asarray(x, np.float32), training=False)).ravel()

    def predict(self, source, filename: str | None = None, include_spectrogram: bool = False) -> dict:
        """source can be a path, raw bytes or a file object."""
        self.reload_if_changed()
        t0 = time.perf_counter()
        x, meta = audio_to_model_input(source)
        t1 = time.perf_counter()
        probs = self.predict_tensor(x)
        t2 = time.perf_counter()

        p = float(probs.mean())  # long recordings: average the 10 s windows
        label = "abnormal" if p >= self.threshold else "normal"
        result = {
            "filename": filename or (str(source) if isinstance(source, (str, Path)) else None),
            "label": label,
            "status": "Faulty - schedule inspection" if label == "abnormal" else "Healthy",
            "probability_abnormal": round(p, 4),
            "confidence": round(p if label == "abnormal" else 1 - p, 4),
            "threshold": self.threshold,
            "window_probabilities": [round(float(v), 4) for v in probs],
            "model_version": self.version,
            "preprocess_ms": round((t1 - t0) * 1000, 2),
            "inference_ms": round((t2 - t1) * 1000, 2),
            **meta,
        }
        if include_spectrogram:
            mel = x[0, :, :, 0]
            lo, hi = np.percentile(mel, [2, 99.5])
            scaled = np.clip((mel - lo) / max(hi - lo, 1e-6), 0, 1)
            # small uint8 image for the UI, flipped so high frequencies are on top
            result["spectrogram"] = (scaled[::-1] * 255).astype(np.uint8).tolist()
        return result


_default: Predictor | None = None
_default_lock = threading.Lock()


def get_predictor() -> Predictor:
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = Predictor()
    return _default


def predict_file(path: str | Path, **kwargs) -> dict:
    return get_predictor().predict(Path(path), filename=Path(path).name, **kwargs)


def predict_many(paths) -> list[dict]:
    predictor = get_predictor()
    return [predictor.predict(Path(p), filename=Path(p).name) for p in paths]
