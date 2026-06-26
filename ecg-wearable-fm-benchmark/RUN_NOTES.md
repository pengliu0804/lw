# Run Notes

## 2026-06-26 PTB-XL Smoke Run

Command:

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 40 `
  --epochs 1 `
  --batch-size 8 `
  --d-model 48 `
  --layers 1 `
  --heads 4 `
  --lead-dropout 0.25 `
  --noise-std 0.02 `
  --baseline-wander 0.03
```

Result:

- Dataset: PTB-XL 100 Hz subset
- Records: 40
- Train/test: 30/10
- Device: CPU
- Label counts: NORM=8, MI=20, STTC=14, CD=14, HYP=14
- Final macro AUROC: 0.7256
- Final macro AUPRC: 0.7249
- Final macro F1: 0.4691
- Artifacts: `runs/20260626-151913/`

Interpretation:

This validates the end-to-end benchmark path: PTB-XL metadata parsing, waveform
download, diagnostic superclass labels, official fold-aware split, wearable-style
augmentations, Transformer training, and multilabel ECG metrics. The sample is
too small for scientific claims.

Next run:

```powershell
.\.venv\Scripts\python.exe -m ecg_wearable_fm.train `
  --max-records 500 `
  --epochs 10 `
  --batch-size 32 `
  --d-model 96 `
  --layers 2 `
  --heads 4 `
  --lead-dropout 0.25 `
  --noise-std 0.02 `
  --baseline-wander 0.03
```

Next research steps:

- Add a clean baseline without augmentations.
- Add single-lead and 3-lead evaluation curves.
- Add ECG-FM or ECGFounder frozen embeddings.
- Add adapter/LoRA style parameter-efficient tuning.
