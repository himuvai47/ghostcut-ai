from __future__ import annotations

from io import BytesIO
import math
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageFilter

from .config import IndexerConfig
from .ffmpeg_tools import extract_frame_bytes
from .models import TimeRange


def _visual_array(data: bytes) -> np.ndarray:
    # Blur + aggressive downsample reduces false change scores from tiny hand/product motion.
    with Image.open(BytesIO(data)) as im:
        im = im.convert("L").filter(ImageFilter.GaussianBlur(radius=1.5)).resize((64, 36))
        return np.asarray(im, dtype=np.float32) / 255.0


def _histogram(arr: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(arr, bins=24, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    total = float(hist.sum()) or 1.0
    return hist / total


def image_difference(jpeg_a: bytes, jpeg_b: bytes) -> float:
    """Robust low-resolution appearance distance in [0,1].

    Pixel MAE catches composition changes; histogram distance makes the score less sensitive
    to small translations/hand movement than the v1 grayscale-pixel-only metric.
    """
    a = _visual_array(jpeg_a)
    b = _visual_array(jpeg_b)
    pixel = float(np.mean(np.abs(a - b)))
    hist = float(np.sum(np.abs(_histogram(a) - _histogram(b))) / 2.0)
    return max(0.0, min(1.0, 0.60 * pixel + 0.40 * hist))


def _even_split(segment: TimeRange, max_duration: float, min_duration: float) -> list[TimeRange]:
    if segment.duration <= max_duration:
        return [segment]
    parts = max(2, math.ceil(segment.duration / max_duration))
    step = segment.duration / parts
    out = [TimeRange(segment.start + i * step, segment.start + (i + 1) * step) for i in range(parts)]
    if len(out) >= 2 and out[-1].duration < min_duration:
        out[-2] = TimeRange(out[-2].start, out[-1].end)
        out.pop()
    return out


def select_change_boundaries(
    shot: TimeRange,
    probe_times: list[float],
    scores: list[float],
    cfg: IndexerConfig,
) -> list[float]:
    """Select conservative semantic-change peaks from consecutive probe scores.

    `scores[i]` is the change between probe_times[i] and probe_times[i+1].
    v1 split on every threshold crossing. v1.1 instead requires an adaptive local peak,
    minimum spacing, edge room, and a per-shot split budget.
    """
    if not scores:
        return []
    adaptive = max(cfg.semantic_diff_threshold, float(median(scores)) + cfg.semantic_adaptive_margin)
    candidates: list[tuple[float, float]] = []  # (score, boundary)
    for i, score in enumerate(scores):
        left = scores[i - 1] if i > 0 else -1.0
        right = scores[i + 1] if i + 1 < len(scores) else -1.0
        if score < adaptive or score < left or score < right:
            continue
        boundary = (probe_times[i] + probe_times[i + 1]) / 2.0
        if boundary - shot.start < cfg.min_semantic_clip_sec:
            continue
        if shot.end - boundary < cfg.min_semantic_clip_sec:
            continue
        candidates.append((score, boundary))

    # A long shot needs some subdivision merely to stay bounded, but semantic peaks beyond
    # that are capped. This prevents a moving hand/product from producing 8-10 tiny clips.
    minimum_boundaries = max(0, math.ceil(shot.duration / cfg.max_semantic_clip_sec) - 1)
    budget = minimum_boundaries + cfg.semantic_extra_split_allowance
    if budget <= 0:
        return []

    selected: list[float] = []
    for _score, boundary in sorted(candidates, reverse=True):
        if len(selected) >= budget:
            break
        if all(abs(boundary - b) >= cfg.semantic_min_boundary_spacing_sec for b in selected):
            selected.append(boundary)
    return sorted(selected)


def _nearest_probe_bytes(target: float, probe_times: list[float], probe_frames: list[bytes]) -> bytes:
    idx = min(range(len(probe_times)), key=lambda i: abs(probe_times[i] - target))
    return probe_frames[idx]


def _merge_visually_redundant(
    segments: list[TimeRange],
    probe_times: list[float],
    probe_frames: list[bytes],
    cfg: IndexerConfig,
) -> list[TimeRange]:
    if len(segments) < 2:
        return segments
    out: list[TimeRange] = [segments[0]]
    for current in segments[1:]:
        previous = out[-1]
        if previous.duration + current.duration > cfg.max_semantic_clip_sec:
            out.append(current)
            continue
        a = _nearest_probe_bytes((previous.start + previous.end) / 2.0, probe_times, probe_frames)
        b = _nearest_probe_bytes((current.start + current.end) / 2.0, probe_times, probe_frames)
        if image_difference(a, b) <= cfg.semantic_merge_diff_threshold:
            out[-1] = TimeRange(previous.start, current.end)
        else:
            out.append(current)
    return out


def subdivide_shot(path: Path, shot: TimeRange, cfg: IndexerConfig) -> list[TimeRange]:
    if shot.duration <= cfg.max_semantic_clip_sec:
        return [shot]

    start = shot.start + min(0.25, shot.duration * 0.05)
    end = shot.end - min(0.25, shot.duration * 0.05)
    probe_times: list[float] = []
    t = start
    while t <= end + 1e-6:
        probe_times.append(min(t, end))
        t += cfg.semantic_probe_interval_sec
    if not probe_times:
        return _even_split(shot, cfg.max_semantic_clip_sec, cfg.min_semantic_clip_sec)
    if probe_times[-1] < end - 0.1:
        probe_times.append(end)

    probe_frames = [extract_frame_bytes(path, t, max_width=384) for t in probe_times]
    scores = [image_difference(a, b) for a, b in zip(probe_frames, probe_frames[1:])]
    changes = select_change_boundaries(shot, probe_times, scores, cfg)

    boundaries = [shot.start] + changes + [shot.end]
    provisional = [TimeRange(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]

    # First merge false semantic changes whose actual surrounding content remains similar.
    provisional = _merge_visually_redundant(provisional, probe_times, probe_frames, cfg)

    # Then enforce a bounded maximum clip duration. These are safe temporal windows rather
    # than claims of additional semantic change; Agent 2 may later trim within them.
    result: list[TimeRange] = []
    for seg in provisional:
        result.extend(_even_split(seg, cfg.max_semantic_clip_sec, cfg.min_semantic_clip_sec))
    return result
