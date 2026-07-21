# xMAE local reproduction starter

This directory contains a downloaded snapshot of the official xMAE code, the
paper PDF, and a small uv-managed smoke test.

## Contents

- `upstream/`: official source from <https://github.com/hzhou3/xMAE>
- `paper/xMAE_ICML_2026.pdf`: arXiv 2605.00973
- `upstream/utils/model_arch/xmae.py`: official model and forward pass
- `upstream/pretrain.py`: official full-pretraining launcher
- `upstream/utils/helper_trainer.py`: official masking, loss, and optimizer logic
- `pyproject.toml` and `uv.lock`: reproducible environment
- `smoke_test.py`: CPU model construction, ECG masking, cross-attention,
  reconstruction loss, and PPG-only inference
- `train_demo.py`: local forward/loss/backward/AdamW/checkpoint demo
- `run_smoke.cmd`: one-command Windows forward runner
- `run_train_demo.cmd`: one-command local training demo

## First run

From the workspace root:

```powershell
.\xmae-reproduction\run_smoke.cmd
```

Manual equivalent:

```powershell
cd .\xmae-reproduction
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --frozen
uv run --frozen python smoke_test.py
```

The smoke test uses synthetic ECG/PPG-like signals and randomly initialized
weights. It validates execution, not scientific accuracy.

## Minimal training run

```powershell
.\xmae-reproduction\run_train_demo.cmd
```

This runs two CPU optimization steps through the official xMAE model and saves
`runs/xmae_demo_checkpoint.pt`. It demonstrates the training mechanics only;
it does not reproduce the paper's pretrained model or metrics.

## Optional notebook environment

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --extra notebook
uv run jupyter lab
```

## Official full-training limitations

The official README says the released `.pth` and `.h5` artifacts contain
made-up weights/data. The official launcher hard-codes a placeholder W&B key
and expects 27 PulseDB HDF5 shards through placeholder S3 settings; the
configured local `h5_path` is not consumed by that launcher. The official full
pretraining command is therefore not an out-of-the-box local path. Files under
`upstream/` are preserved unchanged; the wrappers provide verified model and
training-mechanics entry points.
