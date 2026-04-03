from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


LABEL_TO_INDEX: Dict[str, int] = {
    "FAKE": 0,
    "REAL": 1,
}
INDEX_TO_LABEL: Dict[int, str] = {value: key for key, value in LABEL_TO_INDEX.items()}


@dataclass
class VideoRecord:
    subset: str
    split: str
    method: str
    label: str
    relative_path: str
    video_id: str
    source_id: str
    target_id: str
    frame_dir: str
    num_frames: int
    source_fps: float = 0.0
    target_fps: float = 0.0
    sample_mode: str = ""

    @property
    def label_index(self) -> int:
        return LABEL_TO_INDEX[self.label.upper()]


def _safe_float(value: str) -> float:
    try:
        return float(value) if value not in ("", None) else 0.0
    except ValueError:
        return 0.0


def load_metadata(
    metadata_csv: Path,
    splits: Optional[Sequence[str]] = None,
    subset: Optional[str] = None,
) -> List[VideoRecord]:
    metadata_csv = metadata_csv.resolve()
    records: List[VideoRecord] = []

    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if splits and row["split"] not in splits:
                continue
            if subset and row["subset"] != subset:
                continue

            records.append(
                VideoRecord(
                    subset=row["subset"],
                    split=row["split"],
                    method=row["method"],
                    label=row["label"].upper(),
                    relative_path=row["relative_path"],
                    video_id=row["video_id"],
                    source_id=row.get("source_id", "") or "",
                    target_id=row.get("target_id", "") or "",
                    frame_dir=row["frame_dir"],
                    num_frames=int(row["num_frames"]),
                    source_fps=_safe_float(row.get("source_fps", "")),
                    target_fps=_safe_float(row.get("target_fps", "")),
                    sample_mode=row.get("sample_mode", "") or "",
                )
            )
    return records


def list_frame_paths(data_root: Path, record: VideoRecord) -> List[Path]:
    frame_dir = data_root / record.frame_dir
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found under: {frame_dir}")
    return frame_paths


def repeat_to_length(indices: List[int], target_len: int) -> List[int]:
    if not indices:
        raise ValueError("Cannot repeat empty indices list.")
    while len(indices) < target_len:
        indices.extend(indices[: target_len - len(indices)])
    return indices[:target_len]


def sample_train_indices(num_frames: int, seq_len: int) -> List[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_frames >= seq_len:
        start = random.randint(0, num_frames - seq_len)
        return list(range(start, start + seq_len))
    return repeat_to_length(list(range(num_frames)), seq_len)


def center_clip_indices(num_frames: int, seq_len: int) -> List[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_frames >= seq_len:
        start = max(0, (num_frames - seq_len) // 2)
        return list(range(start, start + seq_len))
    return repeat_to_length(list(range(num_frames)), seq_len)


def build_eval_windows(
    num_frames: int,
    seq_len: int,
    stride: int,
    max_windows: int = 0,
) -> List[List[int]]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    if num_frames <= seq_len:
        return [repeat_to_length(list(range(num_frames)), seq_len)]

    stride = max(1, stride)
    starts = list(range(0, num_frames - seq_len + 1, stride))
    if starts[-1] != num_frames - seq_len:
        starts.append(num_frames - seq_len)

    windows = [list(range(start, start + seq_len)) for start in starts]
    if max_windows > 0 and len(windows) > max_windows:
        keep_idx = np.linspace(0, len(windows) - 1, num=max_windows)
        keep_idx = np.unique(np.round(keep_idx).astype(int)).tolist()
        windows = [windows[i] for i in keep_idx]
    return windows


def load_clip_tensor(
    frame_paths: Sequence[Path],
    indices: Sequence[int],
    transform: Callable[[Image.Image], torch.Tensor],
) -> torch.Tensor:
    frames = []
    for index in indices:
        image = Image.open(frame_paths[index]).convert("RGB")
        frames.append(transform(image))
    return torch.stack(frames, dim=0)


class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        records: Sequence[VideoRecord],
        seq_len: int,
        transform: Callable[[Image.Image], torch.Tensor],
        train: bool = True,
    ) -> None:
        self.data_root = data_root.resolve()
        self.records = list(records)
        self.seq_len = seq_len
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        frame_paths = list_frame_paths(self.data_root, record)

        if self.train:
            clip_indices = sample_train_indices(len(frame_paths), self.seq_len)
        else:
            clip_indices = center_clip_indices(len(frame_paths), self.seq_len)

        clip = load_clip_tensor(frame_paths, clip_indices, self.transform)
        label = torch.tensor(record.label_index, dtype=torch.long)
        return clip, label


def summarize_records(records: Iterable[VideoRecord]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for record in records:
        key = f"{record.split}:{record.label.lower()}"
        summary[key] = summary.get(key, 0) + 1
    return summary
