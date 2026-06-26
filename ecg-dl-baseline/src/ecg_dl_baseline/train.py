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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ecg_dl_baseline.data import load_heartbeat_dataset
from ecg_dl_baseline.model import ECGCnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small ECG 1D-CNN baseline.")
    parser.add_argument("--records", nargs="+", default=["100", "101", "200", "207", "208"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--max-beats", type=int, default=2000)
    parser.add_argument("--max-per-class", type=int, default=700)
    parser.add_argument("--seed", type=int, default=7)
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


def make_loaders(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    indices = np.arange(len(y))
    class_counts = np.bincount(y)
    stratify = y if class_counts[class_counts > 0].min(initial=0) >= 2 else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=seed,
        stratify=stratify,
    )

    x_tensor = torch.from_numpy(x[:, None, :])
    y_tensor = torch.from_numpy(y)
    train_ds = TensorDataset(x_tensor[train_idx], y_tensor[train_idx])
    test_ds = TensorDataset(x_tensor[test_idx], y_tensor[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
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
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * len(batch_y)
        total_items += len(batch_y)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        targets.append(batch_y.cpu().numpy())
    return total_loss / max(total_items, 1), np.concatenate(targets), np.concatenate(preds)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sample_beats(x: np.ndarray, y: np.ndarray, class_names: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(len(class_names), 1, figsize=(8, 8), sharex=True)
    if len(class_names) == 1:
        axes = [axes]
    time = np.arange(x.shape[1])
    for class_idx, ax in enumerate(axes):
        matches = np.flatnonzero(y == class_idx)
        ax.set_ylabel(class_names[class_idx])
        if len(matches) == 0:
            ax.text(0.5, 0.5, "no sample", transform=ax.transAxes, ha="center")
            continue
        ax.plot(time, x[matches[0]], linewidth=1.0)
        ax.axvline(x.shape[1] // 2, color="tab:red", linestyle="--", linewidth=0.8)
    axes[-1].set_xlabel("Samples")
    fig.suptitle("Example Beat Windows")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = args.project_dir / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    x, y, class_names, counts = load_heartbeat_dataset(
        records=args.records,
        project_dir=args.project_dir,
        window_size=args.window_size,
        max_beats=args.max_beats,
        max_per_class=args.max_per_class,
    )
    print(f"Loaded {len(y)} beats with class counts: {dict(counts)}")

    train_loader, test_loader = make_loaders(x, y, args.batch_size, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGCnn(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        acc = accuracy_score(y_true, y_pred)
        bacc = balanced_accuracy_score(y_true, y_pred)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "accuracy": acc,
                "balanced_accuracy": bacc,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"test_loss={test_loss:.4f} acc={acc:.3f} bacc={bacc:.3f}"
        )

    test_loss, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names = [class_names[i] for i in present_labels]
    report = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "args": vars(args),
        },
        run_dir / "model.pt",
    )
    plot_confusion_matrix(cm, class_names, run_dir / "confusion_matrix.png")
    plot_sample_beats(x, y, class_names, run_dir / "sample_beats.png")

    metrics = {
        "records": args.records,
        "num_beats": int(len(y)),
        "class_counts": dict(counts),
        "device": str(device),
        "history": history,
        "final_accuracy": float(accuracy_score(y_true, y_pred)),
        "final_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Saved run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
