# ECG DL Baseline

A small open-source-style baseline for deep learning on ECG heartbeat segments.

It downloads a small subset of the MIT-BIH Arrhythmia Database through WFDB, cuts
beat-centered ECG windows, trains a compact 1D CNN in PyTorch, and writes metrics
and figures to `runs/`.

## Why this project

This is meant as a first runnable ECG deep learning sandbox before moving to
lab data such as ECG+SCG/GCG, radar, PPG, or multimodal vital-sign signals.

Default task:

- Dataset: MIT-BIH Arrhythmia Database
- Input: lead/channel 0 ECG windows centered at annotated beats
- Label: coarse AAMI-style beat class
- Model: compact 1D CNN
- Split: stratified beat-level split for smoke testing

For publishable work, replace the split with subject-level or record-level
validation. Beat-level random splits are only for quick code verification.

## Setup

From the workspace root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\ecg-dl-baseline\requirements.txt
```

## Smoke run

```powershell
.\.venv\Scripts\python.exe -m ecg_dl_baseline.train `
  --records 100 101 200 207 208 `
  --epochs 2 `
  --max-beats 2000 `
  --batch-size 64
```

Outputs:

- `ecg-dl-baseline/data/raw/mitdb/`: downloaded WFDB records
- `ecg-dl-baseline/runs/<timestamp>/metrics.json`
- `ecg-dl-baseline/runs/<timestamp>/model.pt`
- `ecg-dl-baseline/runs/<timestamp>/confusion_matrix.png`
- `ecg-dl-baseline/runs/<timestamp>/sample_beats.png`

## Verified run

On 2026-06-26, the smoke command above completed on CPU with MIT-BIH records
100, 101, 200, 207, and 208.

- Loaded beats: 1953
- Class counts: N=700, S=175, V=700, F=374, Q=4
- Epochs: 2
- Accuracy: 0.8875
- Balanced accuracy: 0.5821
- Artifacts: `runs/20260626-143401/`

The high accuracy is inflated by class imbalance and beat-level splitting; use
the balanced accuracy and confusion matrix to judge the smoke run. For a real
paper or lab report, move to record-level or subject-level validation.

## Next experiments

- Change split to record-level validation.
- Add ECG morphology features or CWT/STFT baselines.
- Replace the CNN with TCN, Transformer, or self-supervised pretraining.
- Add signal-quality labels and motion/noise augmentation.
- Extend input channels to ECG+SCG/GCG or ECG+PPG.
