from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .config import IndexerConfig
from .ffmpeg_tools import extract_frame_file
from .models import TimeRange
from .utils import clamp


def frame_count_for_duration(duration: float, cfg: IndexerConfig) -> int:
    if duration <= 3.0:
        return cfg.short_clip_frame_count
    if duration <= 8.0:
        return cfg.normal_clip_frame_count
    if duration <= 16.0:
        return cfg.long_clip_frame_count
    return cfg.very_long_clip_frame_count


def representative_times(segment: TimeRange, cfg: IndexerConfig) -> list[float]:
    count = max(1, frame_count_for_duration(segment.duration, cfg))
    pad = min(cfg.frame_edge_pad_sec, max(0.0, segment.duration * 0.12))
    low = segment.start + pad
    high = segment.end - pad
    if high <= low:
        return [round((segment.start + segment.end) / 2.0, 6)]
    if count == 1:
        return [round((low + high) / 2.0, 6)]
    # Sample inside the range rather than on exact boundaries.
    step = (high - low) / (count + 1)
    return [round(clamp(low + step * (i + 1), segment.start, segment.end), 6) for i in range(count)]


def extract_representative_frames(
    source: Path,
    segment: TimeRange,
    clip_uid: str,
    source_uid: str,
    frames_root: Path,
    cfg: IndexerConfig,
) -> tuple[list[float], list[Path]]:
    times = representative_times(segment, cfg)
    clip_dir = frames_root / source_uid / clip_uid
    clip_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx, t in enumerate(times, start=1):
        path = clip_dir / f"frame_{idx:02d}_{t:.3f}s.jpg"
        if not path.exists() or path.stat().st_size == 0:
            extract_frame_file(source, t, path, max_width=cfg.frame_max_width)
        paths.append(path)
    return times, paths


def is_nearly_black(paths: list[Path], mean_threshold: float, std_threshold: float) -> bool:
    if not paths:
        return False
    means: list[float] = []
    stds: list[float] = []
    for path in paths:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L").resize((128, 72)), dtype=np.float32) / 255.0
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
    return max(means) <= mean_threshold and max(stds) <= std_threshold
