from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torchvision import transforms

from dataset import (
    VideoRecord,
    build_eval_windows,
    list_frame_paths,
    load_metadata,
    load_clip_tensor,
)
from model import build_model


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_eval_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def compute_binary_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float32)
    y_pred = (y_score >= threshold).astype(np.int64)

    metrics: Dict[str, float] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) >= 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = float("nan")
    return metrics


def find_best_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> Tuple[float, Dict[str, float]]:
    best_threshold = 0.5
    best_metrics = compute_binary_metrics(y_true, y_score, threshold=best_threshold)

    for threshold in np.arange(0.1, 0.91, 0.02):
        metrics = compute_binary_metrics(y_true, y_score, threshold=float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


@torch.inference_mode()
def predict_video_score(
    model: torch.nn.Module,
    record: VideoRecord,
    data_root: Path,
    transform: transforms.Compose,
    seq_len: int,
    stride: int,
    max_windows: int,
    device: torch.device,
    clip_batch_size: int = 8,
) -> Tuple[float, int]:
    frame_paths = list_frame_paths(data_root, record)
    windows = build_eval_windows(len(frame_paths), seq_len=seq_len, stride=stride, max_windows=max_windows)

    scores: List[float] = []
    pending: List[torch.Tensor] = []
    for window in windows:
        pending.append(load_clip_tensor(frame_paths, window, transform))
        if len(pending) >= clip_batch_size:
            batch = torch.stack(pending, dim=0).to(device, non_blocking=True)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 0]
            scores.extend(probs.cpu().tolist())
            pending.clear()

    if pending:
        batch = torch.stack(pending, dim=0).to(device, non_blocking=True)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)[:, 0]
        scores.extend(probs.cpu().tolist())

    return float(np.mean(scores)), len(windows)


@torch.inference_mode()
def evaluate_records(
    model: torch.nn.Module,
    records: Sequence[VideoRecord],
    data_root: Path,
    seq_len: int,
    img_size: int,
    stride: int,
    max_windows: int,
    device: torch.device,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
    clip_batch_size: int = 8,
) -> Dict[str, object]:
    transform = build_eval_transform(img_size)
    y_true: List[int] = []
    y_score: List[float] = []
    predictions: List[Dict[str, object]] = []

    for idx, record in enumerate(records, start=1):
        score, num_windows = predict_video_score(
            model=model,
            record=record,
            data_root=data_root,
            transform=transform,
            seq_len=seq_len,
            stride=stride,
            max_windows=max_windows,
            device=device,
            clip_batch_size=clip_batch_size,
        )
        target = 1 if record.label.upper() == "FAKE" else 0
        y_true.append(target)
        y_score.append(score)
        predictions.append(
            {
                "index": idx,
                "video_id": record.video_id,
                "split": record.split,
                "method": record.method,
                "label": record.label,
                "fake_score": score,
                "num_windows": num_windows,
                "frame_dir": record.frame_dir,
            }
        )

    if optimize_threshold:
        threshold, metrics = find_best_threshold(y_true, y_score)
    else:
        metrics = compute_binary_metrics(y_true, y_score, threshold=threshold)

    return {
        "metrics": metrics,
        "predictions": predictions,
        "threshold": metrics["threshold"],
    }


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    model = build_model(
        backbone_name=config.get("backbone_name", "xception"),
        pretrained=False,
        hidden_dim=int(config.get("hidden_dim", 256)),
        lstm_layers=int(config.get("lstm_layers", 1)),
        dropout=float(config.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def resolve_subset(split: str, subset: Optional[str]) -> Optional[str]:
    if subset is None or subset == "":
        return "external_test" if split == "external_test" else "main"
    if subset not in {"main", "external_test"}:
        raise ValueError(f"Unsupported subset: {subset}")
    return subset


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
    parser = argparse.ArgumentParser(description="Evaluate deepfake video model.")
    parser.add_argument("--data-root", type=Path, default=root / "dataset" / "FF")
    parser.add_argument("--metadata", type=Path, default=root / "dataset" / "FF" / "metadata.csv")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "external_test"])
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional metadata subset. Defaults to 'main' for train/val/test and 'external_test' for external_test.",
    )
    parser.add_argument("--img-size", type=int, default=299)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    data_root = resolve_input_path(args.data_root, project_root)
    metadata = resolve_input_path(args.metadata, project_root)
    device = torch.device(args.device)
    model, checkpoint_config = load_checkpoint_model(args.checkpoint.resolve(), device)
    subset = resolve_subset(args.split, args.subset)

    records = load_metadata(metadata, splits=[args.split], subset=subset)
    result = evaluate_records(
        model=model,
        records=records,
        data_root=data_root,
        seq_len=checkpoint_config.get("seq_len", args.seq_len),
        img_size=checkpoint_config.get("img_size", args.img_size),
        stride=args.stride,
        max_windows=args.max_windows,
        device=device,
        threshold=args.threshold,
        optimize_threshold=args.optimize_threshold,
        clip_batch_size=args.clip_batch_size,
    )

    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
