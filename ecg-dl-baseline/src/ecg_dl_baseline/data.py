from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
import requests
import wfdb


AAMI_GROUPS = {
    "N": {"N", "L", "R", "e", "j"},
    "S": {"A", "a", "J", "S"},
    "V": {"V", "E"},
    "F": {"F"},
    "Q": {"/", "f", "Q"},
}

CLASS_NAMES = list(AAMI_GROUPS.keys())
SYMBOL_TO_CLASS = {
    symbol: class_name
    for class_name, symbols in AAMI_GROUPS.items()
    for symbol in symbols
}

PHYSIONET_MITDB_BASE_URL = "https://physionet.org/files/mitdb/1.0.0"
REQUIRED_EXTENSIONS = ("hea", "dat", "atr")


def ensure_mitdb(records: list[str], project_dir: Path) -> Path:
    """Download selected MIT-BIH records if they are not already present."""
    db_dir = project_dir / "data" / "raw" / "mitdb"
    db_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        for extension in REQUIRED_EXTENSIONS:
            out_path = db_dir / f"{record}.{extension}"
            if out_path.exists():
                continue
            url = f"{PHYSIONET_MITDB_BASE_URL}/{record}.{extension}"
            print(f"Downloading {url}")
            _download_file(url, out_path)
    return db_dir


def _download_file(url: str, out_path: Path, retries: int = 3) -> None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as exc:  # pragma: no cover - exercised by flaky networks
            last_error = exc
            if out_path.exists():
                out_path.unlink()
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def _robust_zscore(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    median = np.median(signal)
    mad = np.median(np.abs(signal - median))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = np.std(signal) + 1e-6
    return ((signal - median) / scale).astype(np.float32)


def _segment_zscore(segment: np.ndarray) -> np.ndarray:
    mean = float(segment.mean())
    std = float(segment.std())
    if std < 1e-6:
        std = 1.0
    return ((segment - mean) / std).astype(np.float32)


def load_heartbeat_dataset(
    records: list[str],
    project_dir: Path,
    window_size: int = 256,
    max_beats: int | None = None,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], Counter]:
    """Load beat-centered ECG windows and coarse AAMI labels."""
    if window_size % 2 != 0:
        raise ValueError("window_size must be even")

    db_dir = ensure_mitdb(records, project_dir)
    half = window_size // 2
    segments: list[np.ndarray] = []
    labels: list[int] = []
    counts: Counter = Counter()
    per_class_seen: dict[str, int] = defaultdict(int)

    for record_name in records:
        record_path = db_dir / record_name
        record = wfdb.rdrecord(str(record_path), channels=[0])
        annotation = wfdb.rdann(str(record_path), "atr")
        signal = _robust_zscore(record.p_signal[:, 0])

        for sample, symbol in zip(annotation.sample, annotation.symbol):
            class_name = SYMBOL_TO_CLASS.get(symbol)
            if class_name is None:
                continue
            start = int(sample) - half
            end = int(sample) + half
            if start < 0 or end > len(signal):
                continue
            if max_per_class is not None and per_class_seen[class_name] >= max_per_class:
                continue

            segment = _segment_zscore(signal[start:end])
            segments.append(segment)
            labels.append(CLASS_NAMES.index(class_name))
            counts[class_name] += 1
            per_class_seen[class_name] += 1

            if max_beats is not None and len(segments) >= max_beats:
                break

        if max_beats is not None and len(segments) >= max_beats:
            break

    if not segments:
        raise RuntimeError("No heartbeat segments were loaded. Check records and annotations.")

    present_classes = {CLASS_NAMES[label] for label in labels}
    if len(present_classes) < 2:
        raise RuntimeError(
            "Only one class was loaded. Add arrhythmia-rich records such as 200, 207, or 208."
        )

    x = np.stack(segments).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return x, y, CLASS_NAMES, counts
