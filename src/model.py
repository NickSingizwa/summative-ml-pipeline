"""CNN on log-mel spectrograms: building, training, fine-tuning and evaluation."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import keras

tf.get_logger().setLevel("ERROR")  # hides the retracing warnings
from keras import layers, regularizers
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from . import config as C


def build_cnn(normalizer_data: np.ndarray | None = None, *, filters=(16, 32, 64, 128),
              dropout: float = 0.3, l2: float = 1e-4, batch_norm: bool = True,
              dense_units: int = 64, input_shape=C.INPUT_SHAPE,
              name: str = C.MODEL_NAME) -> keras.Model:
    """4 conv blocks + small dense head. normalizer_data = training spectrograms used to fit the scaler."""
    inputs = keras.Input(shape=input_shape, name="log_mel")
    # scaler lives inside the model so it gets saved with it
    norm = layers.Normalization(axis=1, name="band_norm")  # mean/var per mel band
    if normalizer_data is not None:
        norm.adapt(np.asarray(normalizer_data, dtype=np.float32))
    x = norm(inputs)

    reg = regularizers.l2(l2) if l2 else None
    for i, n_filters in enumerate(filters):
        x = layers.Conv2D(n_filters, 3, padding="same", use_bias=not batch_norm,
                          kernel_regularizer=reg, name=f"conv{i + 1}")(x)
        if batch_norm:
            x = layers.BatchNormalization(name=f"bn{i + 1}")(x)
        x = layers.Activation("relu", name=f"relu{i + 1}")(x)
        if i < len(filters) - 1:
            x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)

    # average over time only, faults show up in specific frequency bands so keep those
    x = layers.AveragePooling2D(pool_size=(1, x.shape[2]), name="time_avg")(x)
    x = layers.Flatten(name="flatten")(x)
    if dropout:
        x = layers.Dropout(dropout, name="drop1")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_regularizer=reg, name="dense")(x)
    if dropout:
        x = layers.Dropout(dropout, name="drop2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_abnormal")(x)
    return keras.Model(inputs, outputs, name=name)


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def _augment(x, y):
    """Random time shift plus one frequency mask and one time mask (like SpecAugment)."""
    n_mels, n_frames = C.N_MELS, C.N_FRAMES
    # pump noise is stationary so shifting in time doesn't change the label
    x = tf.roll(x, shift=tf.random.uniform([], 0, n_frames, dtype=tf.int32), axis=1)
    fill = tf.reduce_mean(x)

    f = tf.random.uniform([], 0, 7, dtype=tf.int32)  # up to 6 mel bands
    f0 = tf.random.uniform([], 0, n_mels - f + 1, dtype=tf.int32)
    fmask = tf.logical_and(tf.range(n_mels) >= f0, tf.range(n_mels) < f0 + f)[:, None, None]

    t = tf.random.uniform([], 0, 21, dtype=tf.int32)  # up to 20 frames
    t0 = tf.random.uniform([], 0, n_frames - t + 1, dtype=tf.int32)
    tmask = tf.logical_and(tf.range(n_frames) >= t0, tf.range(n_frames) < t0 + t)[None, :, None]

    x = tf.where(tf.logical_or(fmask, tmask), fill, x)
    return x, y


def make_dataset(X: np.ndarray, y: np.ndarray, batch_size: int = 32, *, shuffle: bool = False,
                 augment: bool = False, seed: int = C.SEED) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((X.astype(np.float32), y.astype(np.float32)))
    if shuffle:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def class_weights(y: np.ndarray) -> dict[int, float]:
    # abnormal is only ~11% of the data, weight it up so the model doesn't ignore it
    counts = np.bincount(y.astype(int), minlength=2)
    total = counts.sum()
    return {i: float(total / (2 * c)) if c else 1.0 for i, c in enumerate(counts)}


def train_model(model: keras.Model, X_train, y_train, X_val, y_val, *, epochs: int = 60,
                batch_size: int = 32, learning_rate: float = 1e-3, early_stopping: bool = True,
                patience: int = 8, reduce_lr: bool = True, use_class_weights: bool = True,
                augment: bool = True, extra_callbacks=None, verbose: int = 0,
                compile: bool = True) -> keras.callbacks.History:
    if compile:
        compile_model(model, learning_rate)
    callbacks = list(extra_callbacks or [])
    if early_stopping:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True))
    if reduce_lr:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 3), min_lr=1e-5))
    return model.fit(
        make_dataset(X_train, y_train, batch_size, shuffle=True, augment=augment),
        validation_data=make_dataset(X_val, y_val, batch_size),
        epochs=epochs,
        class_weight=class_weights(y_train) if use_class_weights else None,
        callbacks=callbacks,
        shuffle=False,  # already shuffled in make_dataset
        verbose=verbose,
    )


def fine_tune(model: keras.Model, X_train, y_train, X_val, y_val, *,
              epochs: int = C.FINE_TUNE_EPOCHS, learning_rate: float = C.FINE_TUNE_LR,
              extra_callbacks=None, verbose: int = 0) -> keras.callbacks.History:
    """Keep training an existing model with a lower learning rate (used for retraining)."""
    return train_model(model, X_train, y_train, X_val, y_val, epochs=epochs,
                       learning_rate=learning_rate, patience=4,
                       extra_callbacks=extra_callbacks, verbose=verbose)


def predict_proba(model: keras.Model, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    return model.predict(np.asarray(X, dtype=np.float32), batch_size=batch_size, verbose=0).ravel()


def best_threshold(y_true: np.ndarray, probs: np.ndarray, prevalence: float | None = None) -> float:
    """Threshold with the best F1 on validation data.

    prevalence: reweight samples to this share of abnormal clips. Needed when retraining
    because the replay buffer has more faults than real data, which pushes the threshold too low.
    """
    y_true = np.asarray(y_true).astype(int)
    weights = None
    if prevalence is not None and 0 < y_true.mean() < 1:
        pos = y_true.mean()
        weights = np.where(y_true == 1, prevalence / pos, (1 - prevalence) / (1 - pos))
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [f1_score(y_true, probs >= t, sample_weight=weights, zero_division=0) for t in grid]
    best = max(scores)
    candidates = [t for t, s in zip(grid, scores) if s >= best - 1e-9]
    return float(min(candidates, key=lambda t: abs(t - 0.5)))  # on ties stay close to 0.5


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2  # AUCs need both classes
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "loss": float(log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both else None,
        "roc_auc": float(roc_auc_score(y_true, probs)) if both else None,
        "pr_auc": float(average_precision_score(y_true, probs)) if both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    return metrics_from_probs(y, predict_proba(model, X), threshold)


def save_model(model: keras.Model, path: Path, also_h5: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    if also_h5:
        model.save(path.with_suffix(".h5"))
    return path


def load_model(path: Path) -> keras.Model:
    return keras.models.load_model(path, compile=False)


def history_to_dict(history: keras.callbacks.History) -> dict:
    return {k: [float(v) for v in vals] for k, vals in history.history.items()}
