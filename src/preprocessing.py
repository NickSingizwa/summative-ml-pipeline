"""Audio preprocessing: load -> validate -> 10 s windows -> log-mel spectrogram.
Also has the handcrafted features used for EDA and the baseline models."""
from __future__ import annotations

import io
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO, Iterable, Sequence

import librosa
import numpy as np
import soundfile as sf

from . import config as C


class AudioValidationError(ValueError):
    pass


def load_audio(source: str | Path | bytes | BinaryIO, sr: int = C.SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Returns (mono waveform at `sr`, original sample rate)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        y, file_sr = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:  # soundfile throws a few different errors
        raise AudioValidationError(f"could not decode audio ({exc.__class__.__name__}): "
                                   "use WAV, FLAC, OGG or MP3") from exc
    y = y.mean(axis=1)  # stereo -> mono
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(y, dtype=np.float32), int(file_sr)


def validate_audio(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> None:
    if y.size == 0:
        raise AudioValidationError("recording is empty")
    duration = y.size / sr
    if duration < C.MIN_SECONDS:
        raise AudioValidationError(f"recording is {duration:.2f} s; at least {C.MIN_SECONDS:.0f} s is needed")
    if not np.isfinite(y).all():
        raise AudioValidationError("recording contains invalid samples")
    if float(np.max(np.abs(y))) < 1e-4:
        raise AudioValidationError("recording is silent")


def split_into_clips(y: np.ndarray, clip_samples: int = C.CLIP_SAMPLES) -> list[np.ndarray]:
    # short clips get looped instead of zero-padded, pump noise is fairly constant
    # so this looks closer to the training data than silence would
    if y.size <= clip_samples:
        reps = math.ceil(clip_samples / y.size)
        return [np.tile(y, reps)[:clip_samples]]
    clips = []
    for start in range(0, y.size, clip_samples):
        seg = y[start:start + clip_samples]
        if seg.size == clip_samples:
            clips.append(seg)
        elif seg.size >= clip_samples // 2:  # drop leftovers shorter than 5 s
            clips.append(np.tile(seg, 2)[:clip_samples])
    return clips


def log_mel_spectrogram(y: np.ndarray, sr: int = C.SAMPLE_RATE) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX, power=2.0,
    )
    # ref=1.0 keeps the absolute level so the model can still see loudness
    return librosa.power_to_db(mel, ref=1.0, top_db=None).astype(np.float32)


def clips_to_model_input(clips: Sequence[np.ndarray]) -> np.ndarray:
    mels = np.stack([log_mel_spectrogram(c) for c in clips])
    return mels[..., np.newaxis]


def audio_to_model_input(source, sr: int = C.SAMPLE_RATE) -> tuple[np.ndarray, dict]:
    """Everything needed to go from a file to model input, shape (n_clips, 64, 157, 1)."""
    y, file_sr = load_audio(source, sr)
    validate_audio(y, sr)
    clips = split_into_clips(y)
    meta = {"duration_s": round(y.size / sr, 3), "original_sample_rate": file_sr, "n_clips": len(clips)}
    return clips_to_model_input(clips), meta


FEATURE_DESCRIPTIONS = {
    "rms_mean": "Loudness (mean RMS energy)",
    "rms_std": "Loudness fluctuation (RMS std)",
    "spectral_centroid_hz": "Brightness (spectral centroid, Hz)",
    "spectral_bandwidth_hz": "Spread of frequencies (bandwidth, Hz)",
    "spectral_rolloff_hz": "85% energy roll-off (Hz)",
    "spectral_flatness": "Noisiness (spectral flatness)",
    "zero_crossing_rate": "Zero-crossing rate",
    "low_band_energy_ratio": "Share of energy below 500 Hz",
}


def handcrafted_features(y: np.ndarray, sr: int = C.SAMPLE_RATE, n_mfcc: int = 13) -> dict:
    """Summary features per clip, used in the EDA and for the classical models."""
    spec = np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)) ** 2
    mag = np.sqrt(spec)
    rms = librosa.feature.rms(S=mag, frame_length=C.N_FFT)[0]
    centroid = librosa.feature.spectral_centroid(S=mag, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=mag, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=sr, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(S=mag)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=C.N_FFT, hop_length=C.HOP_LENGTH)[0]
    freqs = librosa.fft_frequencies(sr=sr, n_fft=C.N_FFT)
    total = spec.sum() + 1e-12
    mel = librosa.feature.melspectrogram(S=spec, sr=sr, n_mels=C.N_MELS, fmin=C.FMIN, fmax=C.FMAX)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=n_mfcc)

    feats = {
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "spectral_centroid_hz": float(centroid.mean()),
        "spectral_bandwidth_hz": float(bandwidth.mean()),
        "spectral_rolloff_hz": float(rolloff.mean()),
        "spectral_flatness": float(flatness.mean()),
        "zero_crossing_rate": float(zcr.mean()),
        "low_band_energy_ratio": float(spec[freqs < 500].sum() / total),
    }
    for i, row in enumerate(mfcc):
        feats[f"mfcc_{i + 1}_mean"] = float(row.mean())
        feats[f"mfcc_{i + 1}_std"] = float(row.std())
    return feats


_LABEL_PATTERN = re.compile(r"^(abnormal|anomaly|faulty|fault|normal)(?=[_\-\s.]|$)")
_LABEL_ALIASES = {"normal": "normal", "abnormal": "abnormal", "anomaly": "abnormal",
                  "faulty": "abnormal", "fault": "abnormal"}


def infer_label(name: str) -> str | None:
    """Get the label from the folder name (normal/, abnormal/) or a filename prefix."""
    parts = [p for p in re.split(r"[\\/]", name.lower()) if p]
    if not parts:
        return None
    for folder in reversed(parts[:-1]):
        if folder in _LABEL_ALIASES:
            return _LABEL_ALIASES[folder]
    match = _LABEL_PATTERN.match(parts[-1])
    return _LABEL_ALIASES[match.group(1)] if match else None


def list_labelled_files(split_dir: Path) -> tuple[list[Path], np.ndarray]:
    """Files in split_dir/normal and split_dir/abnormal with labels 0/1."""
    paths, labels = [], []
    for cls in C.CLASSES:
        for path in sorted((split_dir / cls).glob("*")):
            if path.suffix.lower() in C.AUDIO_EXTENSIONS:
                paths.append(path)
                labels.append(C.LABEL_TO_INDEX[cls])
    return paths, np.asarray(labels, dtype=np.int32)


def _process_path(path: str, with_features: bool) -> tuple[np.ndarray, dict | None]:
    y, _ = load_audio(path)
    validate_audio(y)
    clip = split_into_clips(y)[0]  # dataset clips are all exactly 10 s
    mel = log_mel_spectrogram(clip)
    feats = handcrafted_features(clip) if with_features else None
    return mel, feats


def extract_dataset(paths: Iterable[Path], with_features: bool = False,
                    n_jobs: int | None = None) -> tuple[np.ndarray, list[dict] | None]:
    # runs in parallel, takes ~15 s for the whole dataset on my laptop
    paths = [str(p) for p in paths]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        results = list(pool.map(_process_path, paths, [with_features] * len(paths), chunksize=16))
    mels = np.stack([r[0] for r in results]).astype(np.float16)[..., np.newaxis]  # float16 halves the size
    feats = [r[1] for r in results] if with_features else None
    return mels, feats
