from __future__ import annotations

import ast
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import wfdb


PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
DIAGNOSTIC_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEAD_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def ensure_ptbxl_metadata(project_dir: Path) -> tuple[Path, Path]:
    data_dir = project_dir / "data" / "ptbxl"
    data_dir.mkdir(parents=True, exist_ok=True)
    database_path = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"

    if not database_path.exists():
        _download_file(f"{PTBXL_BASE_URL}/ptbxl_database.csv", database_path)
    if not scp_path.exists():
        _download_file(f"{PTBXL_BASE_URL}/scp_statements.csv", scp_path)
    return database_path, scp_path


def _download_file(url: str, out_path: Path, retries: int = 3) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url}")
            with requests.get(url, stream=True, timeout=90) as response:
                response.raise_for_status()
                with out_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            if out_path.exists():
                out_path.unlink()
            if attempt < retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}") from last_error


def load_ptbxl_index(project_dir: Path) -> pd.DataFrame:
    database_path, scp_path = ensure_ptbxl_metadata(project_dir)
    database = pd.read_csv(database_path, index_col="ecg_id")
    scp = pd.read_csv(scp_path, index_col=0)
    diagnostic_map = {
        code: row["diagnostic_class"]
        for code, row in scp.iterrows()
        if bool(row.get("diagnostic", 0)) and isinstance(row.get("diagnostic_class"), str)
    }

    labels = []
    for raw_codes in database["scp_codes"]:
        codes = ast.literal_eval(raw_codes)
        row_labels = np.zeros(len(DIAGNOSTIC_CLASSES), dtype=np.float32)
        for code in codes:
            diagnostic_class = diagnostic_map.get(code)
            if diagnostic_class in DIAGNOSTIC_CLASSES:
                row_labels[DIAGNOSTIC_CLASSES.index(diagnostic_class)] = 1.0
        labels.append(row_labels)

    label_array = np.stack(labels)
    for i, class_name in enumerate(DIAGNOSTIC_CLASSES):
        database[class_name] = label_array[:, i]
    database["has_diagnostic_label"] = label_array.sum(axis=1) > 0
    return database[database["has_diagnostic_label"]].copy()


def select_balanced_subset(
    index: pd.DataFrame,
    max_records: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_records is None or len(index) <= max_records:
        return index.sort_index()

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    by_class: dict[str, list[int]] = {}
    per_class_target = max(1, max_records // len(DIAGNOSTIC_CLASSES))

    for class_name in DIAGNOSTIC_CLASSES:
        ids = index.index[index[class_name] > 0].to_numpy()
        rng.shuffle(ids)
        by_class[class_name] = ids.tolist()
        selected.update(by_class[class_name][:per_class_target])

    if len(selected) < max_records:
        remaining = np.array([idx for idx in index.index if idx not in selected])
        rng.shuffle(remaining)
        selected.update(remaining[: max_records - len(selected)].tolist())

    selected_list = sorted(list(selected))[:max_records]
    return index.loc[selected_list].sort_index()


def ensure_waveforms(project_dir: Path, index: pd.DataFrame, workers: int = 8) -> None:
    data_dir = project_dir / "data" / "ptbxl"
    jobs = []
    for relative_record in index["filename_lr"].unique():
        base_path = data_dir / relative_record
        for extension in ("hea", "dat"):
            out_path = base_path.with_suffix(f".{extension}")
            if out_path.exists():
                continue
            url = f"{PTBXL_BASE_URL}/{relative_record}.{extension}"
            jobs.append((url, out_path))
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(_download_file, url, out_path) for url, out_path in jobs]
        for future in as_completed(futures):
            future.result()


def load_waveform(project_dir: Path, relative_record: str) -> np.ndarray:
    data_dir = project_dir / "data" / "ptbxl"
    record_path = data_dir / relative_record
    signal, _ = wfdb.rdsamp(str(record_path))
    signal = signal.astype(np.float32)
    signal = np.nan_to_num(signal)
    return normalize_ecg(signal)


def normalize_ecg(signal: np.ndarray) -> np.ndarray:
    median = np.median(signal, axis=0, keepdims=True)
    mad = np.median(np.abs(signal - median), axis=0, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(signal, axis=0, keepdims=True)
    scale = np.where(scale < 1e-6, std + 1e-6, scale)
    return ((signal - median) / scale).astype(np.float32)


def build_arrays(
    project_dir: Path,
    index: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    waveforms = []
    labels = []
    folds = []
    for _, row in index.iterrows():
        ecg = load_waveform(project_dir, row["filename_lr"])
        waveforms.append(ecg.T)
        labels.append(row[DIAGNOSTIC_CLASSES].to_numpy(dtype=np.float32))
        folds.append(int(row["strat_fold"]))
    return (
        np.stack(waveforms).astype(np.float32),
        np.stack(labels).astype(np.float32),
        np.asarray(folds, dtype=np.int64),
    )


def summarize_labels(labels: np.ndarray) -> dict[str, int]:
    counts = labels.sum(axis=0).astype(int)
    return {class_name: int(counts[i]) for i, class_name in enumerate(DIAGNOSTIC_CLASSES)}
