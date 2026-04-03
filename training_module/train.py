from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import VideoSequenceDataset, load_metadata, summarize_records
from eval import evaluate_records
from model import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_loader(
    data_root: Path,
    records,
    seq_len: int,
    img_size: int,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    dataset = VideoSequenceDataset(
        data_root=data_root,
        records=records,
        seq_len=seq_len,
        transform=build_train_transform(img_size) if train else build_eval_transform(img_size),
        train=train,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def compute_class_weights(records) -> torch.Tensor:
    counts = {"FAKE": 0, "REAL": 0}
    for record in records:
        counts[record.label.upper()] += 1
    fake = max(1, counts["FAKE"])
    real = max(1, counts["REAL"])
    total = fake + real
    weights = torch.tensor([total / (2 * fake), total / (2 * real)], dtype=torch.float32)
    return weights


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_sum = 0.0
    total = 0
    correct = 0

    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
    }


def pick_score(metrics: Dict[str, float]) -> float:
    auc = metrics.get("auc", float("nan"))
    if not math.isnan(auc):
        return auc
    return metrics.get("f1", 0.0)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    config: Dict[str, object],
    metrics: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_input_path(path: Path, project_root: Path) -> Path:
    candidate = path.expanduser()
    if candidate.exists():
        return candidate.resolve()

    # Compatibility for hosted environments where users pass "/dataset/..."
    # but actual files are under "<project_root>/dataset/...".
    if candidate.is_absolute():
        parts = candidate.parts
        if len(parts) >= 2 and parts[1] == "dataset":
            fallback = project_root.joinpath(*parts[1:])
            if fallback.exists():
                return fallback.resolve()

    return candidate.resolve()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train Xception/EfficientNet + BiLSTM deepfake detector.")
    parser.add_argument("--data-root", type=Path, default=root / "dataset" / "FF")
    parser.add_argument("--metadata", type=Path, default=root / "dataset" / "FF" / "metadata.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "training_module" / "outputs" / "xception_lstm")
    parser.add_argument("--backbone-name", type=str, default="xception")
    parser.add_argument("--img-size", type=int, default=299)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--eval-stride", type=int, default=5)
    parser.add_argument("--eval-max-windows", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--external-after-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    project_root = Path(__file__).resolve().parents[1]

    data_root = resolve_input_path(args.data_root, project_root)
    metadata = resolve_input_path(args.metadata, project_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_records = load_metadata(metadata, splits=["train"], subset="main")
    val_records = load_metadata(metadata, splits=["val"], subset="main")
    test_records = load_metadata(metadata, splits=["test"], subset="main")
    external_records = load_metadata(metadata, splits=["external_test"], subset="external_test")

    print("Record summary:", summarize_records(train_records + val_records + test_records))
    print(f"External records: {len(external_records)}")

    train_loader = make_loader(
        data_root=data_root,
        records=train_records,
        seq_len=args.seq_len,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train=True,
    )

    class_weights = compute_class_weights(train_records).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(
        backbone_name=args.backbone_name,
        pretrained=True,
        hidden_dim=args.hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    config = {
        "backbone_name": args.backbone_name,
        "img_size": args.img_size,
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "lstm_layers": args.lstm_layers,
        "dropout": args.dropout,
        "label_order": ["FAKE", "REAL"],
    }

    history: List[Dict[str, object]] = []
    best_score = -1.0
    best_threshold = 0.5
    best_ckpt = output_dir / "best.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_result = evaluate_records(
            model=model,
            records=val_records,
            data_root=data_root,
            seq_len=args.seq_len,
            img_size=args.img_size,
            stride=args.eval_stride,
            max_windows=args.eval_max_windows,
            device=device,
            threshold=0.5,
            optimize_threshold=True,
            clip_batch_size=args.clip_batch_size,
        )
        val_metrics = val_result["metrics"]
        best_threshold = float(val_result["threshold"])
        score = pick_score(val_metrics)
        scheduler.step()

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_auc={val_metrics['auc']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} thr={best_threshold:.2f}"
        )

        save_checkpoint(
            output_dir / "last.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            metrics=epoch_result,
        )

        if score > best_score:
            best_score = score
            save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                config={**config, "best_threshold": best_threshold},
                metrics=epoch_result,
            )

    best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    val_result = evaluate_records(
        model=model,
        records=val_records,
        data_root=data_root,
        seq_len=args.seq_len,
        img_size=args.img_size,
        stride=args.eval_stride,
        max_windows=args.eval_max_windows,
        device=device,
        threshold=best_state.get("config", {}).get("best_threshold", 0.5),
        optimize_threshold=False,
        clip_batch_size=args.clip_batch_size,
    )
    test_result = evaluate_records(
        model=model,
        records=test_records,
        data_root=data_root,
        seq_len=args.seq_len,
        img_size=args.img_size,
        stride=args.eval_stride,
        max_windows=args.eval_max_windows,
        device=device,
        threshold=best_state.get("config", {}).get("best_threshold", 0.5),
        optimize_threshold=False,
        clip_batch_size=args.clip_batch_size,
    )

    summary: Dict[str, object] = {
        "config": best_state.get("config", config),
        "best_val": val_result["metrics"],
        "test": test_result["metrics"],
        "history": history,
    }
    val_predictions_payload = {
        "split": "val",
        "threshold": val_result["threshold"],
        "metrics": val_result["metrics"],
        "predictions": val_result["predictions"],
    }
    test_predictions_payload = {
        "split": "test",
        "threshold": test_result["threshold"],
        "metrics": test_result["metrics"],
        "predictions": test_result["predictions"],
    }

    if args.external_after_train and external_records:
        external_result = evaluate_records(
            model=model,
            records=external_records,
            data_root=data_root,
            seq_len=args.seq_len,
            img_size=args.img_size,
            stride=args.eval_stride,
            max_windows=args.eval_max_windows,
            device=device,
            threshold=best_state.get("config", {}).get("best_threshold", 0.5),
            optimize_threshold=False,
            clip_batch_size=args.clip_batch_size,
        )
        summary["external_test"] = external_result["metrics"]
        save_json(
            output_dir / "external_test_predictions.json",
            {
                "split": "external_test",
                "threshold": external_result["threshold"],
                "metrics": external_result["metrics"],
                "predictions": external_result["predictions"],
            },
        )

    save_json(output_dir / "metrics.json", summary)
    save_json(output_dir / "val_predictions.json", val_predictions_payload)
    save_json(output_dir / "test_predictions.json", test_predictions_payload)

    print("Best val metrics:", json.dumps(summary["best_val"], ensure_ascii=False))
    print("Test metrics:", json.dumps(summary["test"], ensure_ascii=False))
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
