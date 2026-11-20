"""SQLite database: uploaded audio (stored as blobs), retrain runs, prediction log and settings."""
from __future__ import annotations

import io
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

from . import config as C

SCHEMA = """
CREATE TABLE IF NOT EXISTS upload_batches (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    label_mode    TEXT NOT NULL,
    n_files       INTEGER DEFAULT 0,
    n_accepted    INTEGER DEFAULT 0,
    n_rejected    INTEGER DEFAULT 0,
    n_duplicates  INTEGER DEFAULT 0,
    eval_metrics  TEXT,
    eval_version  TEXT
);
CREATE TABLE IF NOT EXISTS uploads (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id      TEXT NOT NULL REFERENCES upload_batches(id),
    filename      TEXT NOT NULL,
    label         TEXT NOT NULL CHECK (label IN ('normal', 'abnormal')),
    sha256        TEXT NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL,
    audio         BLOB NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending',   -- pending / processed / rejected
    features      BLOB,                              -- log-mel windows saved as .npy bytes
    n_windows     INTEGER,
    duration_s    REAL,
    sample_rate   INTEGER,
    error         TEXT,
    uploaded_at   TEXT NOT NULL,
    used_in_run   INTEGER                            -- first retrain run that used it
);
CREATE INDEX IF NOT EXISTS idx_uploads_status ON uploads(status, used_in_run);
CREATE TABLE IF NOT EXISTS training_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    trigger         TEXT NOT NULL,
    reason          TEXT,
    status          TEXT NOT NULL,                   -- running / completed / failed
    stage           TEXT,
    progress        REAL DEFAULT 0,
    message         TEXT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    base_version    TEXT,
    new_version     TEXT,
    promoted        INTEGER,
    n_new_samples   INTEGER,
    n_total_samples INTEGER,
    metrics_before  TEXT,
    metrics_after   TEXT,
    history         TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL,
    filename        TEXT,
    label           TEXT,
    probability     REAL,
    latency_ms      REAL,
    model_version   TEXT,
    host            TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DEFAULT_SETTINGS = {
    "auto_retrain": C.DEFAULT_AUTO_RETRAIN,
    "retrain_threshold": C.DEFAULT_RETRAIN_THRESHOLD,
    "f1_floor": C.DEFAULT_F1_FLOOR,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(C.DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # lets several containers share the db file
        conn.executescript(SCHEMA)
        for key, value in DEFAULT_SETTINGS.items():
            conn.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (key, json.dumps(value)))
        # if the server died during a retrain the run would stay "running" forever
        conn.execute("UPDATE training_runs SET status='failed', message='interrupted (server restarted)', "
                     "finished_at=? WHERE status='running'", (now(),))


# feature (de)serialisation
def array_to_blob(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def blob_to_array(blob: bytes) -> np.ndarray:
    return np.load(io.BytesIO(blob), allow_pickle=False)


# uploads
def create_batch(label_mode: str) -> str:
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    with connect() as conn:
        conn.execute("INSERT INTO upload_batches(id, created_at, label_mode) VALUES (?, ?, ?)",
                     (batch_id, now(), label_mode))
    return batch_id


def add_upload(batch_id: str, filename: str, label: str, audio: bytes, sha256: str) -> int | None:
    """Returns the new row id, or None if this exact file was already uploaded (same sha256)."""
    with connect() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO uploads(batch_id, filename, label, sha256, size_bytes, audio, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (batch_id, filename, label, sha256, len(audio), audio, now()))
        return cur.lastrowid if cur.rowcount else None


def pending_uploads() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT id, filename, label, audio FROM uploads WHERE status='pending'").fetchall()


def mark_processed(upload_id: int, features: np.ndarray, duration_s: float, sample_rate: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE uploads SET status='processed', features=?, n_windows=?, duration_s=?, "
                     "sample_rate=?, error=NULL WHERE id=?",
                     (array_to_blob(features.astype(np.float16)), int(features.shape[0]),
                      duration_s, sample_rate, upload_id))


def mark_rejected(upload_id: int, error: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE uploads SET status='rejected', error=? WHERE id=?", (error[:300], upload_id))


def finish_batch(batch_id: str, n_files: int, n_accepted: int, n_rejected: int, n_duplicates: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE upload_batches SET n_files=?, n_accepted=?, n_rejected=?, n_duplicates=? WHERE id=?",
                     (n_files, n_accepted, n_rejected, n_duplicates, batch_id))


def set_batch_eval(batch_id: str, metrics: dict, version: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE upload_batches SET eval_metrics=?, eval_version=? WHERE id=?",
                     (json.dumps(metrics), version, batch_id))


def processed_features(batch_id: str | None = None, only_new: bool = False):
    """Returns (ids, X, y). A long recording gives one sample per 10 s window."""
    sql = "SELECT id, label, features FROM uploads WHERE status='processed'"
    args: list = []
    if batch_id:
        sql += " AND batch_id=?"
        args.append(batch_id)
    if only_new:
        sql += " AND used_in_run IS NULL"
    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    ids, xs, ys = [], [], []
    for row in rows:
        feats = blob_to_array(row["features"])
        ids.append(row["id"])
        xs.append(feats)
        ys.extend([C.LABEL_TO_INDEX[row["label"]]] * len(feats))
    if not xs:
        return [], np.empty((0, *C.INPUT_SHAPE), np.float16), np.empty((0,), np.int32)
    return ids, np.concatenate(xs), np.asarray(ys, np.int32)


def mark_used(upload_ids: list[int], run_id: int) -> None:
    with connect() as conn:
        conn.executemany("UPDATE uploads SET used_in_run=? WHERE id=? AND used_in_run IS NULL",
                         [(run_id, i) for i in upload_ids])


def upload_summary() -> dict:
    with connect() as conn:
        counts = conn.execute(
            "SELECT label, status, (used_in_run IS NOT NULL) AS used, COUNT(*) AS n "
            "FROM uploads GROUP BY label, status, used").fetchall()
        batches = conn.execute("SELECT * FROM upload_batches ORDER BY created_at DESC LIMIT 20").fetchall()
    summary = {"total": 0, "pending_training": 0, "used_in_training": 0, "rejected": 0,
               "by_label": {"normal": 0, "abnormal": 0}}
    for row in counts:
        summary["total"] += row["n"]
        if row["status"] == "rejected":
            summary["rejected"] += row["n"]
            continue
        summary["by_label"][row["label"]] += row["n"]
        if row["used"]:
            summary["used_in_training"] += row["n"]
        else:
            summary["pending_training"] += row["n"]
    summary["batches"] = [{**dict(b), "eval_metrics": json.loads(b["eval_metrics"]) if b["eval_metrics"] else None}
                          for b in batches]
    return summary


def batch_files(batch_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT id, filename, label, status, n_windows, duration_s, error, used_in_run "
                            "FROM uploads WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
    return [dict(r) for r in rows]


def count_new_processed() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM uploads WHERE status IN ('pending', 'processed') "
                            "AND used_in_run IS NULL").fetchone()[0]


# training runs
_RUN_JSON = ("metrics_before", "metrics_after", "history")


def _run_dict(row) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in _RUN_JSON:
        out[key] = json.loads(out[key]) if out[key] else None
    out["promoted"] = None if out["promoted"] is None else bool(out["promoted"])
    return out


def create_run(trigger: str, reason: str, base_version: str | None) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO training_runs(trigger, reason, status, stage, started_at, base_version) "
                           "VALUES (?, ?, 'running', 'queued', ?, ?)", (trigger, reason, now(), base_version))
        return cur.lastrowid


def update_run(run_id: int, **fields) -> None:
    for key in _RUN_JSON:
        if key in fields and fields[key] is not None and not isinstance(fields[key], str):
            fields[key] = json.dumps(fields[key])
    if "promoted" in fields and fields["promoted"] is not None:
        fields["promoted"] = int(fields["promoted"])
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE training_runs SET {cols} WHERE id=?", (*fields.values(), run_id))


def get_run(run_id: int) -> dict | None:
    with connect() as conn:
        return _run_dict(conn.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone())


def latest_run() -> dict | None:
    with connect() as conn:
        return _run_dict(conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 1").fetchone())


def running_run() -> dict | None:
    with connect() as conn:
        return _run_dict(conn.execute(
            "SELECT * FROM training_runs WHERE status='running' ORDER BY id DESC LIMIT 1").fetchone())


def list_runs(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_run_dict(r) for r in rows]


# predictions log
def log_prediction(filename: str | None, label: str, probability: float, latency_ms: float,
                   model_version: str, host: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO predictions(created_at, filename, label, probability, latency_ms, model_version, host) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (now(), filename, label, probability, latency_ms, model_version, host))


def prediction_summary(limit: int = 15) -> dict:
    with connect() as conn:
        totals = conn.execute("SELECT label, COUNT(*) AS n, AVG(latency_ms) AS avg_ms FROM predictions "
                              "GROUP BY label").fetchall()
        recent = conn.execute("SELECT created_at, filename, label, probability, latency_ms, model_version, host "
                              "FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        latencies = [r[0] for r in conn.execute(
            "SELECT latency_ms FROM predictions ORDER BY id DESC LIMIT 500").fetchall()]
    return {
        "total": sum(r["n"] for r in totals),
        "by_label": {r["label"]: r["n"] for r in totals},
        "recent": [dict(r) for r in recent],
        "latency_ms_recent": latencies[::-1],
    }


# settings
def get_settings() -> dict:
    with connect() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    out = dict(DEFAULT_SETTINGS)
    out.update({r["key"]: json.loads(r["value"]) for r in rows})
    return out


def update_settings(**values) -> dict:
    with connect() as conn:
        for key, value in values.items():
            if key in DEFAULT_SETTINGS and value is not None:
                conn.execute("INSERT INTO settings(key, value) VALUES (?, ?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))
    return get_settings()
