"""Paths and settings shared by the notebook, the API and the retraining job.
Paths can be overridden with env vars (used in Docker / HF Spaces)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT_DIR / "models"))

RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
NEW_BATCH_DIR = DATA_DIR / "new_batch"          # kept aside to test retraining
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "pipeline.db"))

TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.npz"
REPLAY_FEATURES_PATH = PROCESSED_DIR / "replay_features.npz"
INSIGHTS_PATH = PROCESSED_DIR / "insights.json"

REGISTRY_PATH = MODELS_DIR / "registry.json"
MODEL_NAME = "pump_sound_cnn"

# audio / spectrogram settings
SAMPLE_RATE = 16_000
CLIP_SECONDS = 10.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
MIN_SECONDS = 1.0
N_FFT = 2048
HOP_LENGTH = 1024
N_MELS = 64
FMIN = 20.0
FMAX = SAMPLE_RATE / 2
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH       # 157 frames for 10 s
INPUT_SHAPE = (N_MELS, N_FRAMES, 1)

CLASSES = ("normal", "abnormal")
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASSES)}
AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3")

SEED = 42

# retraining
FINE_TUNE_LR = float(os.getenv("FINE_TUNE_LR", "2e-4"))
FINE_TUNE_EPOCHS = int(os.getenv("FINE_TUNE_EPOCHS", "15"))
# a new model is only promoted if it doesn't drop more than this on the test set
PROMOTION_PR_AUC_TOLERANCE = float(os.getenv("PROMOTION_PR_AUC_TOLERANCE", "0.005"))
PROMOTION_F1_TOLERANCE = float(os.getenv("PROMOTION_F1_TOLERANCE", "0.03"))
DEFAULT_AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
DEFAULT_RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "100"))  # new clips before auto retrain
DEFAULT_F1_FLOOR = float(os.getenv("F1_FLOOR", "0.80"))                 # retrain if F1 on new data is below this

# api limits
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "600"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(25 * 1024 * 1024)))
MAX_ZIP_BYTES = int(os.getenv("MAX_ZIP_BYTES", str(600 * 1024 * 1024)))
LOG_PREDICTIONS = os.getenv("LOG_PREDICTIONS", "true").lower() == "true"
