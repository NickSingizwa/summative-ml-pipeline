"""Uploads the app to a Hugging Face Docker Space.

    hf auth login
    python scripts/deploy_hf_space.py --space <username>/pumpsense

Only copies what the container needs (code, model v1, processed features, demo files).
HF builds the Dockerfile after the upload.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

INCLUDE = [
    "Dockerfile", ".dockerignore", "requirements.txt", "wheelhouse/.gitkeep",
    "src", "api", "ui",
    "models/pump_sound_cnn.keras",
    "data/processed/test_features.npz", "data/processed/replay_features.npz", "data/processed/insights.json",
    "data/samples", "data/retrain_demo_batch.zip",
]

SPACE_README = """---
title: PumpSense
emoji: 🔧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Water-pump fault detection from 10-second audio clips
---

# PumpSense: water-pump fault detection from sound

Upload a 10-second recording of a pump and a CNN on log-mel spectrograms classifies it as **normal** or
**abnormal**. The dashboard also shows uptime, data insights, bulk upload of labelled audio, and one-click
retraining (the current model is fine-tuned and only promoted if it does better on the held-out test set).

* Dashboard: this Space's app page
* API docs: `/docs`
* Source code, notebook and load-test results: {github}

Data: DCASE 2020 Challenge Task 2 / MIMII pump recordings (CC BY 4.0). Uploaded data and retrained models
live in the container's storage and reset when the Space restarts.
"""


def stage(target: Path) -> None:
    for item in INCLUDE:
        src = ROOT / item
        if not src.exists():
            raise SystemExit(f"missing {item}: run the notebook first")
        dst = target / item
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        else:
            shutil.copy2(src, dst)
    # only ship v1 from the notebook, not versions from local retrain tests
    reg = json.loads(C.REGISTRY_PATH.read_text())
    v1 = reg["versions"]["v1"]
    (target / "models" / "registry.json").write_text(json.dumps({"current": "v1", "versions": {"v1": v1}}, indent=2))
    (target / "models" / "versions").mkdir(parents=True, exist_ok=True)
    (target / "models" / "versions" / ".gitkeep").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", required=True, help="<hf-username>/<space-name>")
    parser.add_argument("--github", default="https://github.com/NickSingizwa/summative-ml-pipeline")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        stage(staging)
        (staging / "README.md").write_text(SPACE_README.format(github=args.github))
        api.upload_folder(folder_path=staging, repo_id=args.space, repo_type="space",
                          commit_message="Deploy PumpSense", delete_patterns=["*"])
    owner, name = args.space.split("/")
    print(f"Space:  https://huggingface.co/spaces/{args.space}")
    print(f"App:    https://{owner.lower()}-{name.lower().replace('_', '-')}.hf.space  (ready once the build finishes)")


if __name__ == "__main__":
    main()
