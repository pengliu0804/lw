from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_wearable_fm.augment import apply_train_augmentations, keep_leads
from ecg_wearable_fm.model import PatchTransformerECG
from ecg_wearable_fm.ptbxl import (
    DIAGNOSTIC_CLASSES,
    LEAD_NAMES,
    build_arrays,
    ensure_waveforms,
    load_ptbxl_index,
    select_balanced_subset,
    summarize_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PTB-XL wearable ECG benchmark.")
    parser.add_argument("--max-records", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--lead-dropout", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--baseline-wander", type=float, default=0.03)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--eval-leads", nargs="*", default=None, help="Optional lead names, e.g. I II V1")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Project directory containing data/ and runs/.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_split(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(folds != 10)
    test_idx = np.flatnonzero(folds == 10)
    if len(test_idx) >= 10 and len(train_idx) >= 10:
        return train_idx, test_idx

    indices = np.arange(len(y))
    multilabel_stratify = np.argmax(y, axis=1)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=multilabel_stratify,
    )
    return train_idx, test_idx


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_idx, test_idx = make_split(x, y, folds, seed)
    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)
    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    test_ds = TensorDataset(x_tensor[test_idx], y_tensor[test_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_idx, test_idx


def make_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    positives = y_train.sum(axis=0)
    negatives = len(y_train) - positives
    weights = negatives / np.maximum(positives, 1)
    return torch.tensor(weights, dtype=torch.float32)


def parse_eval_leads(eval_leads: list[str] | None) -> list[int] | None:
    if not eval_leads:
        return None
    lookup = {lead.upper(): idx for idx, lead in enumerate(LEAD_NAMES)}
    return [lookup[lead.upper()] for lead in eval_leads if lead.upper() in lookup]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    lead_dropout: float,
    noise_std: float,
    baseline_wander: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_x = apply_train_augmentations(batch_x, lead_dropout, noise_std, baseline_wander)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_y)
        total_items += len(batch_y)
    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    eval_lead_indices: list[int] | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_targets = []
    all_probs = []
    for batch_x, batch_y in loader:
        batch_x = keep_leads(batch_x.to(device), eval_lead_indices)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        probs = torch.sigmoid(logits)
        total_loss += float(loss.item()) * len(batch_y)
        total_items += len(batch_y)
        all_targets.append(batch_y.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return total_loss / max(total_items, 1), np.concatenate(all_targets), np.concatenate(all_probs)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(np.int64)
    per_class = {}
    aucs = []
    auprcs = []
    f1s = []
    for class_idx, class_name in enumerate(DIAGNOSTIC_CLASSES):
        true_col = y_true[:, class_idx]
        prob_col = y_prob[:, class_idx]
        pred_col = y_pred[:, class_idx]
        class_metrics = {
            "support": int(true_col.sum()),
            "f1": float(f1_score(true_col, pred_col, zero_division=0)),
        }
        f1s.append(class_metrics["f1"])
        if len(np.unique(true_col)) == 2:
            class_metrics["auroc"] = float(roc_auc_score(true_col, prob_col))
            class_metrics["auprc"] = float(average_precision_score(true_col, prob_col))
            aucs.append(class_metrics["auroc"])
            auprcs.append(class_metrics["auprc"])
        else:
            class_metrics["auroc"] = None
            class_metrics["auprc"] = None
        per_class[class_name] = class_metrics

    return {
        "macro_auroc": float(np.mean(aucs)) if aucs else None,
        "macro_auprc": float(np.mean(auprcs)) if auprcs else None,
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class,
    }


def plot_auc(metrics: dict, out_path: Path) -> None:
    names = []
    values = []
    for class_name, class_metrics in metrics["per_class"].items():
        if class_metrics["auroc"] is not None:
            names.append(class_name)
            values.append(class_metrics["auroc"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color="#3874a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC")
    ax.set_title("PTB-XL AUROC by Diagnostic Class")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sample_ecg(x: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    sample = x[0]
    label_text = ", ".join(
        class_name for class_idx, class_name in enumerate(DIAGNOSTIC_CLASSES) if y[0, class_idx] > 0
    )
    fig, axes = plt.subplots(12, 1, figsize=(10, 9), sharex=True)
    time = np.arange(sample.shape[1]) / 100.0
    for lead_idx, ax in enumerate(axes):
        ax.plot(time, sample[lead_idx], linewidth=0.8)
        ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=18)
    axes[-1].set_xlabel("Seconds")
    fig.suptitle(f"PTB-XL Sample ECG ({label_text})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    index = load_ptbxl_index(args.project_dir)
    index = select_balanced_subset(index, max_records=args.max_records, seed=args.seed)
    ensure_waveforms(args.project_dir, index, workers=args.download_workers)
    x, y, folds = build_arrays(args.project_dir, index)

    run_dir = args.project_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_sample_ecg(x, y, run_dir / "sample_ecg.png")

    train_loader, test_loader, train_idx, test_idx = make_loaders(x, y, folds, args.batch_size, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchTransformerECG(
        num_classes=len(DIAGNOSTIC_CLASSES),
        d_model=args.d_model,
        num_layers=args.layers,
        nhead=args.heads,
    ).to(device)
    pos_weight = make_pos_weight(y[train_idx]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    eval_lead_indices = parse_eval_leads(args.eval_leads)

    print(f"Loaded PTB-XL subset: x={x.shape}, label_counts={summarize_labels(y)}")
    print(f"Train={len(train_idx)} Test={len(test_idx)} Device={device}")
    if eval_lead_indices is not None:
        print(f"Evaluation keeps leads: {args.eval_leads}")

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.lead_dropout,
            args.noise_std,
            args.baseline_wander,
        )
        test_loss, y_true, y_prob = evaluate(model, test_loader, criterion, device, eval_lead_indices)
        metrics = compute_metrics(y_true, y_prob)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss, **metrics})
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
            f"macro_auroc={metrics['macro_auroc']} macro_f1={metrics['macro_f1']:.3f}"
        )

    test_loss, y_true, y_prob = evaluate(model, test_loader, criterion, device, eval_lead_indices)
    final_metrics = compute_metrics(y_true, y_prob)
    plot_auc(final_metrics, run_dir / "roc_auc_by_class.png")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": DIAGNOSTIC_CLASSES,
            "lead_names": LEAD_NAMES,
            "args": vars(args),
        },
        run_dir / "model.pt",
    )

    payload = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "num_records": int(len(x)),
        "train_records": int(len(train_idx)),
        "test_records": int(len(test_idx)),
        "label_counts": summarize_labels(y),
        "device": str(device),
        "history": history,
        "final": final_metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
