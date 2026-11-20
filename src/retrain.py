"""Retraining: store uploads -> preprocess -> fine-tune the current model -> compare on the
test set -> promote if it's not worse.

Can be started from the dashboard, POST /api/retrain, automatically after an upload
(enough new clips, or the model doing badly on them), or from the command line:
    python -m src.retrain --ingest data/new_batch
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import os
import threading
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn.model_selection import train_test_split

from . import config as C
from . import database as db
from . import registry
from .preprocessing import AudioValidationError, audio_to_model_input, infer_label

LOCK_PATH = C.MODELS_DIR / ".retrain.lock"


class RetrainBusy(RuntimeError):
    pass


# 1. store uploads
def _iter_audio_entries(filename: str, data: bytes):
    """Yields (name, bytes) for a single audio file or every audio file in a zip."""
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            total = sum(i.file_size for i in zf.infolist())
            if total > C.MAX_ZIP_BYTES:
                raise ValueError(f"zip expands to {total / 1e6:.0f} MB, limit is {C.MAX_ZIP_BYTES / 1e6:.0f} MB")
            for info in zf.infolist():
                path = PurePosixPath(info.filename)
                if info.is_dir() or path.name.startswith(".") or "__MACOSX" in path.parts:
                    continue
                if path.suffix.lower() in C.AUDIO_EXTENSIONS and info.file_size <= C.MAX_FILE_BYTES:
                    # keep just folder/name, enough to read the label and avoids weird paths
                    yield "/".join(path.parts[-2:]), zf.read(info)
    elif Path(filename).suffix.lower() in C.AUDIO_EXTENSIONS:
        yield filename, data


def ingest_files(files: list[tuple[str, bytes]], label_mode: str = "auto") -> dict:
    """Saves the uploaded files in the db.

    label_mode is 'normal'/'abnormal' for the whole batch, or 'auto' to take it from the
    folder or file name (normal_*, abnormal_*, anomaly_*).
    """
    if label_mode not in ("auto", *C.CLASSES):
        raise ValueError("label must be 'auto', 'normal' or 'abnormal'")
    batch_id = db.create_batch(label_mode)
    n_files = accepted = duplicates = 0
    skipped: list[dict] = []
    for filename, data in files:
        for name, audio in _iter_audio_entries(filename, data):
            n_files += 1
            if n_files > C.MAX_UPLOAD_FILES:
                raise ValueError(f"too many files (limit {C.MAX_UPLOAD_FILES})")
            label = label_mode if label_mode != "auto" else infer_label(name)
            if label is None:
                skipped.append({"file": name, "reason": "label unknown: choose a label or name files normal_*/abnormal_*"})
                continue
            if db.add_upload(batch_id, Path(name).name, label, audio, hashlib.sha256(audio).hexdigest()) is None:
                duplicates += 1
            else:
                accepted += 1
    db.finish_batch(batch_id, n_files, accepted, len(skipped), duplicates)
    return {"batch_id": batch_id, "files_received": n_files, "stored": accepted,
            "duplicates": duplicates, "skipped": skipped}


# 2. preprocess
def preprocess_pending() -> dict:
    """Turns every pending upload into log-mel features (saved back in the db)."""
    ok, rejected = 0, []
    for row in db.pending_uploads():
        try:
            x, meta = audio_to_model_input(row["audio"])
            db.mark_processed(row["id"], x, meta["duration_s"], meta["original_sample_rate"])
            ok += 1
        except AudioValidationError as exc:
            db.mark_rejected(row["id"], str(exc))
            rejected.append({"file": row["filename"], "reason": str(exc)})
    return {"processed": ok, "rejected": rejected}


def evaluate_batch(batch_id: str) -> dict | None:
    """How well does the live model do on the batch that was just uploaded?"""
    from .model import metrics_from_probs
    from .prediction import get_predictor

    _, X, y = db.processed_features(batch_id=batch_id)
    if len(y) == 0:
        return None
    predictor = get_predictor()
    predictor.reload_if_changed()
    probs = np.concatenate([predictor.predict_tensor(X[i:i + 64]) for i in range(0, len(X), 64)])
    metrics = metrics_from_probs(y, probs, predictor.threshold)
    db.set_batch_eval(batch_id, metrics, predictor.version)
    return metrics


def check_triggers(batch_metrics: dict | None = None) -> tuple[bool, str]:
    """Checks if an upload should start a retrain automatically."""
    settings = db.get_settings()
    if not settings["auto_retrain"]:
        return False, "auto-retrain disabled"
    waiting = db.count_new_processed()
    if waiting >= settings["retrain_threshold"]:
        return True, f"{waiting} new samples waiting (threshold {settings['retrain_threshold']})"
    if batch_metrics and batch_metrics["n"] >= 20 and batch_metrics.get("roc_auc") is not None \
            and batch_metrics["f1"] < settings["f1_floor"]:
        return True, f"F1 on new data {batch_metrics['f1']:.2f} below floor {settings['f1_floor']:.2f}"
    return False, f"{waiting}/{settings['retrain_threshold']} new samples; no drift detected"


# 3-5. fine-tune, evaluate, promote
def _acquire_lock():
    # file lock instead of a threading lock so it also works across containers
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK_PATH, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        raise RetrainBusy("a retraining run is already in progress")
    return fh


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return data["X"], data["y"]


def promotion_decision(before: dict, after: dict) -> tuple[bool, str]:
    """Compare old vs new on the test set. PR-AUC doesn't depend on the threshold so it's the main check."""
    summary = (f"test PR-AUC {before['pr_auc']:.3f} → {after['pr_auc']:.3f}, "
               f"F1 {before['f1']:.3f} → {after['f1']:.3f}, recall {before['recall']:.3f} → {after['recall']:.3f}")
    if after["pr_auc"] < before["pr_auc"] - C.PROMOTION_PR_AUC_TOLERANCE:
        return False, f"candidate PR-AUC dropped ({summary})"
    if after["f1"] < before["f1"] - C.PROMOTION_F1_TOLERANCE:
        return False, f"candidate F1 dropped ({summary})"
    if after["pr_auc"] >= before["pr_auc"] and after["f1"] >= before["f1"]:
        return True, f"better on the test set ({summary})"
    return True, (f"as good within tolerance: PR-AUC -{C.PROMOTION_PR_AUC_TOLERANCE}, "
                  f"F1 -{C.PROMOTION_F1_TOLERANCE} allowed ({summary})")


def run_retraining(run_id: int) -> dict:
    """Runs one retrain. Expects the retrain lock to be held."""
    import keras
    from .model import best_threshold, evaluate_model, fine_tune, history_to_dict, load_model, predict_proba, save_model

    try:
        db.update_run(run_id, stage="preprocessing", progress=0.05, message="preprocessing uploaded audio")
        pre = preprocess_pending()

        db.update_run(run_id, stage="loading data", progress=0.1)
        new_ids, X_new_only, _ = db.processed_features(only_new=True)
        if not new_ids:
            raise ValueError("no new uploaded samples since the last retrain - upload data first")
        _, X_up, y_up = db.processed_features()  # every upload so far, not just the new ones
        # mix in some of the original training data so the model doesn't forget it
        X_rep, y_rep = _load_npz(C.REPLAY_FEATURES_PATH)
        X = np.concatenate([X_rep, X_up]).astype(np.float32)
        y = np.concatenate([y_rep, y_up]).astype(np.int32)
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=C.SEED)

        reg = registry.load_registry()
        base_entry = registry.current_entry(reg)
        base_model = load_model(registry.model_path(base_entry))
        model = load_model(registry.model_path(base_entry))  # start from the current model, not from scratch

        class Progress(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                db.update_run(run_id, stage="fine-tuning", progress=0.15 + 0.7 * (epoch + 1) / C.FINE_TUNE_EPOCHS,
                              message=f"epoch {epoch + 1}/{C.FINE_TUNE_EPOCHS} - loss {logs.get('loss', 0):.4f}, "
                                      f"val_loss {logs.get('val_loss', 0):.4f}, val_auc {logs.get('val_auc', 0):.4f}")

        db.update_run(run_id, stage="fine-tuning", progress=0.15,
                      n_new_samples=int(len(X_new_only)), n_total_samples=int(len(X)),
                      message=f"fine-tuning {base_entry['version']} on {len(X_tr)} samples "
                              f"({len(X_new_only)} new, {len(X_up)} uploaded in total)")
        history = fine_tune(model, X_tr, y_tr, X_val, y_val, extra_callbacks=[Progress()])

        db.update_run(run_id, stage="evaluating", progress=0.88, message="evaluating on held-out test set")
        X_test, y_test = _load_npz(C.TEST_FEATURES_PATH)
        before = evaluate_model(base_model, X_test, y_test, base_entry["threshold"])
        prevalence = base_entry.get("prevalence") or 0.109
        threshold = best_threshold(y_val, predict_proba(model, X_val), prevalence=prevalence)
        after = evaluate_model(model, X_test, y_test, threshold)

        promoted, reason = promotion_decision(before, after)
        new_version = None
        if promoted:
            new_version = registry.next_version(registry.load_registry())
            filename = f"versions/{C.MODEL_NAME}_{new_version}.keras"
            save_model(model, C.MODELS_DIR / filename)
            registry.register(filename=filename, test_metrics=after, threshold=threshold, trigger="retrain",
                              parent=base_entry["version"], history=history_to_dict(history),
                              train_samples=int(len(X_tr)), prevalence=prevalence, version=new_version,
                              notes=f"run {run_id}: fine-tuned on {len(X_new_only)} new samples")
            message = f"promoted {new_version}: {reason}"
        else:
            message = f"kept {base_entry['version']}: {reason}"
        db.mark_used(new_ids, run_id)
        db.update_run(run_id, status="completed", stage="done", progress=1.0, finished_at=db.now(),
                      new_version=new_version, promoted=promoted, metrics_before=before,
                      metrics_after=after, history=history_to_dict(history), message=message)
        return {"run_id": run_id, "promoted": promoted, "new_version": new_version, "message": message,
                "metrics_before": before, "metrics_after": after, "preprocessing": pre}
    except Exception as exc:
        db.update_run(run_id, status="failed", stage="failed", finished_at=db.now(),
                      message=f"{exc.__class__.__name__}: {exc}")
        raise


def start_retraining(trigger: str = "manual", reason: str = "", background: bool = True) -> int:
    """Starts a retrain (in a background thread by default) and returns the run id."""
    lock = _acquire_lock()
    try:
        base = registry.load_registry().get("current")
        run_id = db.create_run(trigger, reason, base)
    except Exception:
        lock.close()
        raise

    def _job():
        try:
            run_retraining(run_id)
        except Exception:
            pass  # failure already recorded on the run row
        finally:
            lock.close()

    if background:
        threading.Thread(target=_job, name=f"retrain-{run_id}", daemon=True).start()
    else:
        _job()
    return run_id


def ingest_and_maybe_retrain(files: list[tuple[str, bytes]], label_mode: str = "auto") -> dict:
    """What happens on upload: store, preprocess, score the batch, then check the auto triggers."""
    result = ingest_files(files, label_mode)
    pre = preprocess_pending()
    result["processed"] = pre["processed"]
    result["rejected"] = pre["rejected"]
    result["batch_metrics"] = evaluate_batch(result["batch_id"])
    should, reason = check_triggers(result["batch_metrics"])
    result["trigger_check"] = reason
    result["auto_retrain_run"] = None
    if should:
        try:
            result["auto_retrain_run"] = start_retraining("auto", reason)
        except RetrainBusy:
            result["trigger_check"] += " (a run is already in progress)"
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Upload a folder of labelled recordings and retrain the model.")
    parser.add_argument("--ingest", type=Path, help="folder with normal/ and abnormal/ sub-folders (or labelled filenames)")
    parser.add_argument("--label", default="auto", choices=["auto", *C.CLASSES])
    parser.add_argument("--limit", type=int, default=0, help="only ingest this many files (0 = all)")
    args = parser.parse_args()

    db.init_db()
    if args.ingest:
        paths = sorted(p for p in args.ingest.rglob("*") if p.suffix.lower() in C.AUDIO_EXTENSIONS)
        if args.limit:
            rng = np.random.default_rng(C.SEED)
            paths = list(rng.choice(paths, size=min(args.limit, len(paths)), replace=False))
        files = [(f"{p.parent.name}/{p.name}", p.read_bytes()) for p in paths]
        summary = ingest_files(files, args.label)
        print(f"stored {summary['stored']} new recordings ({summary['duplicates']} duplicates)")
    run_id = start_retraining("manual", "command line", background=False)
    run = db.get_run(run_id)
    print(f"run {run_id}: {run['status']} - {run['message']}")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    _cli()
