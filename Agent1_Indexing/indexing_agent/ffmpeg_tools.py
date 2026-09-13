from __future__ import annotations

from io import BytesIO
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from .models import MediaInfo, TimeRange


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str], *, capture_stdout: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
        creationflags=creationflags,
    )


def require_binaries() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FFmpegError(
            "FFmpeg/FFprobe were not found in PATH. Install FFmpeg and reopen PowerShell."
        )
    return ffmpeg, ffprobe


def _parse_fraction(value: str | None) -> float:
    if not value or value in {"0/0", "N/A"}:
        return 0.0
    if "/" in value:
        a, b = value.split("/", 1)
        try:
            denom = float(b)
            return float(a) / denom if denom else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_media(path: Path) -> MediaInfo:
    _, ffprobe = require_binaries()
    cmd = [
        ffprobe, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    proc = _run(cmd, timeout=120)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="replace").strip())
    payload = json.loads(proc.stdout.decode("utf-8", errors="replace"))
    streams = payload.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise FFmpegError("No video stream found")
    fmt = payload.get("format", {})
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    if duration <= 0:
        raise FFmpegError("Could not determine a positive video duration")
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            try:
                rotation = int(side["rotation"])
            except (TypeError, ValueError):
                pass
    tags = video.get("tags") or {}
    if "rotate" in tags:
        try:
            rotation = int(tags["rotate"])
        except (TypeError, ValueError):
            pass
    return MediaInfo(
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_parse_fraction(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        codec=str(video.get("codec_name") or "unknown"),
        format_name=str(fmt.get("format_name") or ""),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        rotation=rotation,
    )


def detect_scene_cuts(path: Path, threshold: float, duration: float) -> list[float]:
    ffmpeg, _ = require_binaries()
    # showinfo reports pts_time for frames selected by FFmpeg's scene-change score.
    vf = f"select=gt(scene\\,{threshold}),showinfo"
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-v", "info", "-i", str(path),
        "-an", "-vf", vf, "-f", "null", "-",
    ]
    proc = _run(cmd, timeout=max(180, int(duration * 5)))
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise FFmpegError(stderr.strip())
    times: list[float] = []
    for match in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", stderr):
        t = float(match.group(1))
        if 0.0 < t < duration:
            times.append(t)
    return sorted(set(round(t, 6) for t in times))


def scene_cuts_to_shots(cuts: Iterable[float], duration: float, min_shot_sec: float) -> list[TimeRange]:
    accepted: list[float] = [0.0]
    for cut in sorted(float(x) for x in cuts):
        if cut - accepted[-1] >= min_shot_sec and duration - cut >= 0.05:
            accepted.append(cut)
    accepted.append(duration)

    shots = [TimeRange(accepted[i], accepted[i + 1]) for i in range(len(accepted) - 1)]
    if len(shots) >= 2 and shots[-1].duration < min_shot_sec:
        prev = shots[-2]
        shots[-2] = TimeRange(prev.start, shots[-1].end)
        shots.pop()
    return [s for s in shots if s.duration > 0.05]


def _scale_filter(max_width: int) -> str:
    # Escape comma inside min() for FFmpeg filter parser.
    return f"scale=min(iw\\,{int(max_width)}):-2"


def extract_frame_bytes(path: Path, timestamp: float, max_width: int = 640) -> bytes:
    ffmpeg, _ = require_binaries()
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{max(0.0, timestamp):.6f}", "-i", str(path),
        "-frames:v", "1", "-vf", _scale_filter(max_width),
        "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1",
    ]
    proc = _run(cmd, timeout=60)
    if proc.returncode != 0 or not proc.stdout:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="replace").strip() or "Frame extraction failed")
    return bytes(proc.stdout)


def extract_frame_file(path: Path, timestamp: float, output: Path, max_width: int = 1280) -> None:
    ffmpeg, _ = require_binaries()
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-ss", f"{max(0.0, timestamp):.6f}", "-i", str(path),
        "-frames:v", "1", "-vf", _scale_filter(max_width),
        "-q:v", "2", str(output),
    ]
    proc = _run(cmd, timeout=60)
    if proc.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="replace").strip() or "Frame extraction failed")
