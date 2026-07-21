"""Minimal local training loop around the official xMAE model.

Synthetic paired signals are used only to verify masking, loss, backward,
AdamW updates, and checkpoint saving on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from smoke_test import make_anchor_mask_ids, make_synthetic_pair


PROJECT_DIR = Path(__file__).resolve().parent


def build_model() -> torch.nn.Module:
    from utils.model_arch.xmae import build_model_from_cfg

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
    return build_model_from_cfg(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny local xMAE training demo.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.9)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs" / "xmae_demo_checkpoint.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("--steps and --batch-size must be positive")

    torch.manual_seed(77)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cpu")
    model = build_model().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    losses: list[float] = []
    for step in range(1, args.steps + 1):
        ppg, ecg = make_synthetic_pair(
            batch_size=args.batch_size,
            length=1000,
            sample_rate=100,
        )
        ppg = (ppg + 0.01 * torch.randn_like(ppg)).to(device)
        ecg = (ecg + 0.01 * torch.randn_like(ecg)).to(device)
        ids_keep_ecg, ids_restore_ecg = make_anchor_mask_ids(
            batch_size=args.batch_size,
            num_patches=25,
            mask_ratio=args.mask_ratio,
            device=device,
        )

        outputs = model(
            ppg=ppg,
            ecg=ecg,
            ids_keep_ppg=None,
            ids_restore_ppg=None,
            ids_keep_ecg=ids_keep_ecg,
            ids_restore_ecg=ids_restore_ecg,
        )
        reconstructed = outputs["ecg_reconstructed"]
        sequence_mask = outputs["seq_mask_used_ecg"]
        loss = (((reconstructed - ecg.unsqueeze(1)) ** 2) * sequence_mask).sum()
        loss = loss / sequence_mask.sum().clamp_min(1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach())
        losses.append(loss_value)
        print(f"step={step:02d} masked_ecg_mse={loss_value:.6f}")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "steps": args.steps,
            "losses": losses,
            "mask_ratio_requested": args.mask_ratio,
            "visible_ecg_patches": int(ids_keep_ecg.shape[1]),
        },
        output_path,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "training": "forward + masked MSE + backward + AdamW step",
                "steps": args.steps,
                "losses": losses,
                "checkpoint": str(output_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

