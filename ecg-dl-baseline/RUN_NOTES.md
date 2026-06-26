# Run Notes

## 2026-06-26 Smoke Run

Command:

```powershell
.\.venv\Scripts\python.exe -m ecg_dl_baseline.train `
  --records 100 101 200 207 208 `
  --epochs 2 `
  --max-beats 2000 `
  --batch-size 64
```

Environment:

- Device: CPU
- PyTorch: 2.12.1+cpu
- WFDB: 4.2.0
- Dataset: MIT-BIH records 100, 101, 200, 207, 208

Result:

- Loaded beats: 1953
- Class counts: N=700, S=175, V=700, F=374, Q=4
- Final accuracy: 0.8875
- Final balanced accuracy: 0.5821
- Artifacts: `runs/20260626-143401/`

Interpretation:

This run validates the end-to-end pipeline: download, heartbeat segmentation,
1D-CNN training, metric export, model checkpointing, and figures. It is not a
research-grade evaluation because it uses a beat-level random split and has a
severe class imbalance, especially for Q and S.

Immediate next improvements:

- Switch to record-level validation.
- Drop or merge classes with too few examples for a smoke benchmark.
- Add class-balanced sampling or class-weighted loss.
- Add stronger baselines: CWT/STFT + XGBoost, TCN, and contrastive pretraining.
