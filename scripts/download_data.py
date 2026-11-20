"""Downloads the DCASE 2020 Task 2 pump data (~1 GB) and splits it into train/test/new_batch.

    python scripts/download_data.py

Split is stratified by label and pump id: 75% train, 15% test, 10% new_batch
(new_batch is kept aside to test uploading + retraining). Data is CC BY 4.0.
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

URL = "https://zenodo.org/api/records/3678171/files/dev_data_pump.zip/content"
ZIP_PATH = C.RAW_DIR / "dev_data_pump.zip"
MD5 = "90e7091ef722b7238a7f1009365779cd"
NAME_RE = re.compile(r"^(normal|anomaly)_id_(\d+)_(\d+)\.wav$")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and md5sum(ZIP_PATH) == MD5:
        print(f"[ok] {ZIP_PATH.name} already downloaded, checksum verified")
        return
    print(f"[..] downloading {URL} (1.03 GB)")

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and blocks % 2000 == 0:
            print(f"     {done / 1e6:7.0f} / {total / 1e6:.0f} MB", flush=True)

    urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=hook)
    if md5sum(ZIP_PATH) != MD5:
        raise SystemExit("checksum mismatch - delete data/raw/dev_data_pump.zip and retry")
    print("[ok] download complete, checksum verified")


def extract() -> Path:
    out = C.RAW_DIR / "pump"
    if out.exists() and any(out.rglob("*.wav")):
        print(f"[ok] already extracted to {out}")
        return out
    print("[..] extracting")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for info in zf.infolist():
            target = (C.RAW_DIR / info.filename).resolve()
            if not str(target).startswith(str(C.RAW_DIR.resolve())):  # don't extract outside data/raw
                raise SystemExit(f"unsafe path in zip: {info.filename}")
        zf.extractall(C.RAW_DIR)
    return out


def build_splits(raw: Path) -> None:
    rows = []
    for source in ("train", "test"):
        for path in sorted((raw / source).glob("*.wav")):
            m = NAME_RE.match(path.name)
            if not m:
                continue
            kind, pump_id, num = m.groups()
            label = "normal" if kind == "normal" else "abnormal"
            # train/ and test/ in the zip reuse the same numbers, so add a/b to keep names unique
            name = f"{label}_id_{pump_id}_{'a' if source == 'train' else 'b'}{num}.wav"
            rows.append({"src": path, "file": name, "label": label, "pump_id": f"id_{pump_id}",
                         "original_split": source})
    if not rows:
        raise SystemExit(f"no wav files found under {raw}")

    strata = np.array([f"{r['label']}_{r['pump_id']}" for r in rows])
    idx = np.arange(len(rows))
    rest, test = train_test_split(idx, test_size=0.15, stratify=strata, random_state=C.SEED)
    train, new = train_test_split(rest, test_size=0.10 / 0.85, stratify=strata[rest], random_state=C.SEED)
    split_of = {**{i: "train" for i in train}, **{i: "test" for i in test}, **{i: "new_batch" for i in new}}

    for split in ("train", "test", "new_batch"):
        for label in C.CLASSES:
            d = C.DATA_DIR / split / label
            if d.exists():
                for old in d.glob("*.wav"):
                    old.unlink()
            d.mkdir(parents=True, exist_ok=True)

    with open(C.MANIFEST_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "label", "pump_id", "split", "original_split"])
        writer.writeheader()
        for i, r in enumerate(rows):
            split = split_of[i]
            dest = C.DATA_DIR / split / r["label"] / r["file"]
            try:
                dest.hardlink_to(r["src"])  # hardlink so it doesn't take extra space
            except OSError:
                shutil.copy2(r["src"], dest)
            writer.writerow({"path": str(dest.relative_to(C.DATA_DIR)), "label": r["label"],
                             "pump_id": r["pump_id"], "split": split, "original_split": r["original_split"]})

    counts = Counter((split_of[i], r["label"]) for i, r in enumerate(rows))
    print(f"[ok] {len(rows)} clips")
    for split in ("train", "test", "new_batch"):
        print(f"     {split:<10} normal {counts[(split, 'normal')]:>5}   abnormal {counts[(split, 'abnormal')]:>4}")
    print(f"[ok] manifest written to {C.MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    download()
    build_splits(extract())
