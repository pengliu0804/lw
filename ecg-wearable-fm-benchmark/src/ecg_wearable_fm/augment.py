from __future__ import annotations

import math

import torch


def lead_dropout(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    """Randomly zero ECG leads. Input shape: (batch, leads, time)."""
    if drop_prob <= 0:
        return x
    keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > drop_prob
    # Keep at least one lead per sample.
    all_dropped = keep.sum(dim=1, keepdim=True) == 0
    if all_dropped.any():
        random_lead = torch.randint(0, x.shape[1], (x.shape[0],), device=x.device)
        keep[torch.arange(x.shape[0], device=x.device), random_lead, 0] = True
    keep = torch.where(all_dropped, keep, keep)
    return x * keep.float()


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def baseline_wander(x: torch.Tensor, amplitude: float, sample_rate: int = 100) -> torch.Tensor:
    if amplitude <= 0:
        return x
    batch, leads, length = x.shape
    t = torch.arange(length, device=x.device, dtype=x.dtype) / sample_rate
    freq = torch.empty(batch, leads, 1, device=x.device, dtype=x.dtype).uniform_(0.15, 0.5)
    phase = torch.empty(batch, leads, 1, device=x.device, dtype=x.dtype).uniform_(0, 2 * math.pi)
    drift = amplitude * torch.sin(2 * math.pi * freq * t.view(1, 1, -1) + phase)
    return x + drift


def apply_train_augmentations(
    x: torch.Tensor,
    lead_drop_prob: float,
    noise_std: float,
    baseline_wander_amp: float,
) -> torch.Tensor:
    x = lead_dropout(x, lead_drop_prob)
    x = gaussian_noise(x, noise_std)
    x = baseline_wander(x, baseline_wander_amp)
    return x


def keep_leads(x: torch.Tensor, lead_indices: list[int] | None) -> torch.Tensor:
    """Zero all leads except the selected leads, preserving 12-lead tensor shape."""
    if lead_indices is None:
        return x
    mask = torch.zeros(1, x.shape[1], 1, device=x.device, dtype=x.dtype)
    mask[:, lead_indices, :] = 1
    return x * mask
