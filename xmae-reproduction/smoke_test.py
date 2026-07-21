"""Run a CPU-friendly forward pass through the official xMAE architecture.

This intentionally avoids the official checkpoint and HDF5 file because the
authors state that both released artifacts contain made-up data/weights.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = PROJECT_DIR / "upstream"
sys.path.insert(0, str(UPSTREAM_DIR))

from utils.model_arch.xmae import build_model_from_cfg  # noqa: E402


def make_anchor_mask_ids(
    *, batch_size: int, num_patches: int, mask_ratio: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the first patches visible and mask one trailing continuous block."""
    num_masked = int(round(mask_ratio * num_patches))
    num_visible = max(1, num_patches - num_masked)

    patch_mask = torch.ones(batch_size, num_patches, device=device)
    patch_mask[:, :num_visible] = 0.0
    base_ids = torch.arange(num_patches, device=device).expand(batch_size, -1)
    ranking = base_ids + patch_mask * num_patches
    ids_shuffle = torch.argsort(ranking, dim=1)
    ids_keep = ids_shuffle[:, :num_visible]
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    return ids_keep.long(), ids_restore.long()


def make_synthetic_pair(
    *, batch_size: int, length: int, sample_rate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic ECG/PPG-like waves for shape and loss validation."""
    time = torch.arange(length, dtype=torch.float32) / sample_rate
    heart_hz = 1.2

    phase = torch.remainder(time * heart_hz, 1.0)
    r_peak = torch.exp(-0.5 * ((phase - 0.18) / 0.018) ** 2)
    t_wave = 0.3 * torch.exp(-0.5 * ((phase - 0.48) / 0.07) ** 2)
    ecg = r_peak + t_wave - 0.15

    delayed_phase = torch.remainder((time - 0.22) * heart_hz, 1.0)
    ppg = torch.where(
        delayed_phase < 0.35,
        torch.sin(math.pi * delayed_phase / 0.35) ** 2,
        0.25 * torch.exp(-(delayed_phase - 0.35) / 0.25),
    )

    ecg = ecg.repeat(batch_size, 1)
    ppg = ppg.repeat(batch_size, 1)
    return ppg, ecg


def main() -> None:
    torch.manual_seed(77)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")

    config = {
        "sampling_freq": 100,
        "seg_len": 10,
        "source": "ppg+ecg",
        "model_params": {
            "latent_dim": 256,
            "d_model": 256,
            "nhead": 8,
            "depth_ecg": 1,
            "depth_ppg": 2,
            "depth_bridge": 1,
            "stem_ch": 32,
            "dropout": 0.1,
            "use_cross_bridge": True,
            "patch_len": 40,
        },
    }

    model = build_model_from_cfg(config).to(device).eval()
    batch_size = 1
    sequence_length = 1000
    patch_length = 40
    mask_ratio = 0.9
    ppg, ecg = make_synthetic_pair(
        batch_size=batch_size,
        length=sequence_length,
        sample_rate=100,
    )
    ppg = ppg.to(device)
    ecg = ecg.to(device)

    ids_keep_ecg, ids_restore_ecg = make_anchor_mask_ids(
        batch_size=batch_size,
        num_patches=sequence_length // patch_length,
        mask_ratio=mask_ratio,
        device=device,
    )

    with torch.inference_mode():
        outputs = model(
            ppg=ppg,
            ecg=ecg,
            ids_keep_ppg=None,
            ids_restore_ppg=None,
            ids_keep_ecg=ids_keep_ecg,
            ids_restore_ecg=ids_restore_ecg,
        )
        ppg_only_outputs = model(ppg=ppg, ecg=None, return_recon=False)

    reconstructed = outputs["ecg_reconstructed"]
    sequence_mask = outputs["seq_mask_used_ecg"]
    masked_mse = (((reconstructed - ecg.unsqueeze(1)) ** 2) * sequence_mask).sum()
    masked_mse = masked_mse / sequence_mask.sum().clamp_min(1.0)

    expected_wave_shape = (batch_size, 1, sequence_length)
    expected_embedding_shape = (batch_size, 256)
    assert tuple(reconstructed.shape) == expected_wave_shape
    assert tuple(outputs["ppg_embedding"].shape) == expected_embedding_shape
    assert tuple(ppg_only_outputs["ppg_embedding"].shape) == expected_embedding_shape
    assert torch.isfinite(masked_mse)

    report = {
        "status": "ok",
        "device": str(device),
        "torch_version": torch.__version__,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "ppg_embedding_shape": list(outputs["ppg_embedding"].shape),
        "ecg_reconstruction_shape": list(reconstructed.shape),
        "visible_ecg_patches": int(ids_keep_ecg.shape[1]),
        "total_ecg_patches": sequence_length // patch_length,
        "effective_mask_fraction": float(sequence_mask.mean()),
        "masked_mse_untrained_model": float(masked_mse),
        "ppg_only_inference": "ok",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

