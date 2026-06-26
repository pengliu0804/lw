# ECG Wearable Foundation Benchmark

Project goal:

> Build a reproducible ECG foundation-model benchmark that can grow into a
> wearable ECG robustness paper.

The v0 project uses PTB-XL 100 Hz ECG records and trains a compact
Transformer-style encoder with wearable-inspired augmentations:

- lead dropout: simulate 12-lead to fewer wearable leads
- Gaussian noise: simulate sensor noise
- baseline wander: simulate low-frequency drift
- small-label training: simulate lab data scarcity

This is not yet ECG-FM integration. It is the benchmark scaffold that makes ECG
data, labels, splits, metrics, and robustness experiments concrete. Once this is
stable, ECG-FM/ECGFounder can be added as stronger backbones.

If you are new to ECG, leads, or medical signal data, start with
`BEGINNER_GUIDE.md` before reading the code.

## Dataset

PTB-XL v1.0.3:

- 21,799 clinical 12-lead ECG records
- 10 seconds per ECG
- 100 Hz and 500 Hz waveform versions
- recommended stratified folds
- diagnostic superclasses: `NORM`, `MI`, `STTC`, `CD`, `HYP`

This project downloads only `ptbxl_database.csv`, `scp_statements.csv`, and the
selected 100 Hz waveform files needed for a run.

## Setup

From the workspace root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .\ecg-wearable-fm-benchmark
```

## Smoke Run

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 160 `
  --epochs 2 `
  --batch-size 16 `
  --lead-dropout 0.25 `
  --noise-std 0.02 `
  --baseline-wander 0.03
```

Outputs:

- `data/ptbxl/`: downloaded PTB-XL metadata and selected waveform files
- `runs/<timestamp>/metrics.json`
- `runs/<timestamp>/model.pt`
- `runs/<timestamp>/roc_auc_by_class.png`
- `runs/<timestamp>/sample_ecg.png`

The verified first run is documented in `RUN_NOTES.md`.

For a zero-background explanation of ECG, 12 leads, PTB-XL labels, the training
log, and each source file, read `BEGINNER_GUIDE.md`.

## Research Direction

The first real paper-shaped question:

> How robust are ECG foundation-model representations when moving from clean
> 12-lead clinical ECG to wearable-like ECG with missing leads, noise, and few
> labels?

Recommended experiment ladder:

1. PTB-XL supervised baseline with official fold split.
2. Single-lead and 3-lead evaluations.
3. Lead dropout and signal noise robustness curves.
4. ECG-FM/ECGFounder frozen encoder + linear probe.
5. Adapter/LoRA/feature-alignment method for robust wearable adaptation.

## Practical Notes

- `--max-records 40` is for a first smoke test.
- `--max-records 500` is a better CPU-scale development run.
- Full PTB-XL experiments should use all records and preferably a GPU.
- Downloads are cached under `data/ptbxl/`; interrupted runs can be resumed.
- Waveform downloads use `--download-workers 8` by default.
