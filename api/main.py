"""FastAPI app: prediction, upload, retraining and monitoring endpoints, plus the dashboard.

Run with: uvicorn api.main:app --port 8000   (docs at /docs)
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config as C
from src import database as db
from src import registry, retrain
from src.preprocessing import AudioValidationError, infer_label
from src.prediction import get_predictor

UI_DIR = C.ROOT_DIR / "ui"
HOST = socket.gethostname()
STARTED_AT = datetime.now(timezone.utc)
_t0 = time.monotonic()


class Stats:
    """Request counters for this process. With several containers each one has its own."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.by_path: Counter = Counter()
        self.predict_latency: deque = deque(maxlen=1000)

    def record(self, path: str, status: int, ms: float):
        with self.lock:
            self.requests += 1
            self.by_path[path] += 1
            if status >= 500:
                self.errors += 1
            if path == "/api/predict" and status < 400:
                self.predict_latency.append(ms)

    def snapshot(self) -> dict:
        with self.lock:
            lat = sorted(self.predict_latency)
            pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 1)) if lat else (lambda q: None)
            return {"requests": self.requests, "errors": self.errors,
                    "by_path": dict(self.by_path.most_common(12)),
                    "predict_latency_ms": {"count": len(lat), "p50": pct(0.5), "p95": pct(0.95),
                                           "mean": round(sum(lat) / len(lat), 1) if lat else None}}


stats = Stats()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    predictor = get_predictor()
    samples = sorted(C.SAMPLES_DIR.glob("*.wav"))
    if samples:  # librosa compiles stuff on first use, do it now instead of on the first request
        predictor.predict(samples[0])
    yield


app = FastAPI(
    title="Pump Sound Fault Detection API",
    description="Classifies 10-second water-pump recordings as **normal** or **abnormal**, "
                "and supports bulk upload of labelled audio plus model retraining.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def track_requests(request: Request, call_next):
    t = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Served-By"] = HOST
        return response
    finally:
        if not request.url.path.startswith("/static"):
            stats.record(request.url.path, status, (time.perf_counter() - t) * 1000)


# health and monitoring
def _uptime() -> dict:
    seconds = time.monotonic() - _t0
    return {"started_at": STARTED_AT.isoformat(timespec="seconds"), "uptime_seconds": round(seconds, 1)}


@app.get("/health", tags=["monitoring"])
def health():
    predictor = get_predictor()
    predictor.reload_if_changed()
    return {"status": "ok", "host": HOST, "model_version": predictor.version, **_uptime()}


@app.get("/api/stats", tags=["monitoring"])
def api_stats():
    return {"host": HOST, **_uptime(), "process": stats.snapshot(),
            "predictions": db.prediction_summary(), "retrain": db.latest_run()}


# model
def _registry_view() -> dict:
    reg = registry.load_registry()
    versions = sorted(reg["versions"].values(), key=lambda v: int(v["version"].lstrip("v")), reverse=True)
    return {"current": reg["current"], "current_entry": reg["versions"].get(reg["current"]),
            "versions": [{k: v for k, v in e.items() if k != "history"} for e in versions]}


@app.get("/api/model", tags=["model"])
def model_info():
    return _registry_view()


@app.post("/api/model/evaluate", tags=["model"])
async def evaluate_current():
    """Re-run the live model on the held-out test set."""
    import numpy as np
    from src.model import metrics_from_probs

    def _run():
        predictor = get_predictor()
        predictor.reload_if_changed()
        with np.load(C.TEST_FEATURES_PATH) as data:
            X, y = data["X"], data["y"]
        t = time.perf_counter()
        probs = np.concatenate([predictor.predict_tensor(X[i:i + 64]) for i in range(0, len(X), 64)])
        metrics = metrics_from_probs(y, probs, predictor.threshold)
        return {"model_version": predictor.version, "evaluated_at": db.now(),
                "seconds": round(time.perf_counter() - t, 2), "metrics": metrics}

    return await run_in_threadpool(_run)


@app.post("/api/model/{version}/activate", tags=["model"])
def activate_version(version: str):
    try:
        entry = registry.activate(version)
    except KeyError:
        raise HTTPException(404, f"unknown version {version}")
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc))
    get_predictor().reload_if_changed()
    return {"current": entry["version"], "test_metrics": entry["test_metrics"]}


# insights and samples
@app.get("/api/insights", tags=["data"])
def insights():
    if not C.INSIGHTS_PATH.exists():
        raise HTTPException(404, "insights not generated yet - run the notebook")
    return JSONResponse(json.loads(C.INSIGHTS_PATH.read_text()))


def _sample_files() -> dict[str, Path]:
    return {p.name: p for p in sorted(C.SAMPLES_DIR.glob("*.wav"))}


@app.get("/api/samples", tags=["data"])
def list_samples():
    return [{"name": n, "label": infer_label(n), "url": f"/api/samples/{n}"} for n in _sample_files()]


@app.get("/api/samples/{name}", tags=["data"])
def get_sample(name: str):
    path = _sample_files().get(name)
    if path is None:
        raise HTTPException(404, "sample not found")
    return FileResponse(path, media_type="audio/wav", filename=name)


@app.get("/api/demo-batch", tags=["data"])
def demo_batch():
    """Zip of 60 labelled clips that weren't used for training, to try upload + retrain."""
    path = C.DATA_DIR / "retrain_demo_batch.zip"
    if not path.exists():
        raise HTTPException(404, "demo batch not available")
    return FileResponse(path, media_type="application/zip", filename=path.name)


# prediction
@app.post("/api/predict", tags=["prediction"])
async def predict(file: UploadFile = File(..., description="WAV/FLAC/OGG/MP3 recording of a pump (ideally 10 s)"),
                  spectrogram: bool = False):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    if len(data) > C.MAX_FILE_BYTES:
        raise HTTPException(413, f"file larger than {C.MAX_FILE_BYTES // 1_000_000} MB")
    t = time.perf_counter()
    try:
        result = await run_in_threadpool(get_predictor().predict, data, file.filename, spectrogram)
    except AudioValidationError as exc:
        raise HTTPException(422, str(exc))
    result["latency_ms"] = round((time.perf_counter() - t) * 1000, 2)
    result["served_by"] = HOST
    if C.LOG_PREDICTIONS:
        try:
            await run_in_threadpool(db.log_prediction, file.filename, result["label"],
                                    result["probability_abnormal"], result["latency_ms"],
                                    result["model_version"], HOST)
        except Exception:  # a db hiccup shouldn't break the prediction
            pass
    return result


# upload + retraining
@app.post("/api/upload", tags=["retraining"])
async def upload(files: list[UploadFile] = File(..., description="audio files and/or .zip archives"),
                 label: str = Form("auto", description="'normal', 'abnormal' or 'auto' (from folder/filename)")):
    if label not in ("auto", *C.CLASSES):
        raise HTTPException(400, "label must be auto, normal or abnormal")
    payload = []
    for f in files:
        data = await f.read()
        if data:
            payload.append((f.filename or "upload.wav", data))
    if not payload:
        raise HTTPException(400, "no files received")
    try:
        result = await run_in_threadpool(retrain.ingest_and_maybe_retrain, payload, label)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    result["summary"] = db.upload_summary()
    return result


@app.get("/api/uploads", tags=["retraining"])
def uploads():
    return {**db.upload_summary(), "settings": db.get_settings()}


@app.get("/api/uploads/{batch_id}", tags=["retraining"])
def upload_batch(batch_id: str):
    return db.batch_files(batch_id)


class RetrainRequest(BaseModel):
    reason: str = Field("manual trigger from dashboard", max_length=200)


@app.post("/api/retrain", status_code=202, tags=["retraining"])
def trigger_retrain(body: RetrainRequest | None = None):
    if db.count_new_processed() == 0:
        raise HTTPException(409, "no new uploaded data since the last retrain - upload recordings first")
    try:
        run_id = retrain.start_retraining("manual", (body or RetrainRequest()).reason)
    except retrain.RetrainBusy as exc:
        raise HTTPException(409, str(exc))
    return {"run_id": run_id, "status": "started", "status_url": "/api/retrain/status"}


@app.get("/api/retrain/status", tags=["retraining"])
def retrain_status():
    return db.latest_run() or {"status": "never run"}


@app.get("/api/retrain/history", tags=["retraining"])
def retrain_history():
    return db.list_runs()


class SettingsUpdate(BaseModel):
    auto_retrain: bool | None = None
    retrain_threshold: int | None = Field(None, ge=1, le=10_000)
    f1_floor: float | None = Field(None, ge=0, le=1)


@app.get("/api/settings", tags=["retraining"])
def get_settings():
    return db.get_settings()


@app.put("/api/settings", tags=["retraining"])
def put_settings(body: SettingsUpdate):
    return db.update_settings(**body.model_dump())


# web ui
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(UI_DIR / "index.html")
