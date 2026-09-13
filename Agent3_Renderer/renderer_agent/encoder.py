from __future__ import annotations
import subprocess
from .utils import run


def available_encoders() -> str:
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return (p.stdout or "") + (p.stderr or "")
    except Exception:
        return ""


def nvenc_usable() -> bool:
    if "h264_nvenc" not in available_encoders():
        return False
    try:
        p = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:r=30:d=0.1",
                "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-",
            ],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return p.returncode == 0
    except Exception:
        return False


def choose_encoder(cfg) -> str:
    requested = (cfg.encoder or "auto").lower()
    if requested == "auto":
        return "h264_nvenc" if nvenc_usable() else "libx264"
    if requested == "h264_nvenc" and not nvenc_usable():
        raise RuntimeError("h264_nvenc requested but NVENC could not initialize")
    if requested not in {"h264_nvenc", "libx264"}:
        raise ValueError(f"Unsupported encoder: {cfg.encoder}")
    return requested


def video_encode_args(cfg, encoder: str) -> list[str]:
    if encoder == "h264_nvenc":
        return [
            "-c:v", "h264_nvenc",
            "-preset", cfg.nvenc_preset,
            "-cq", str(cfg.nvenc_cq),
            "-b:v", "0",
            "-pix_fmt", cfg.pixel_format,
        ]
    return [
        "-c:v", "libx264",
        "-preset", cfg.x264_preset,
        "-crf", str(cfg.x264_crf),
        "-pix_fmt", cfg.pixel_format,
    ]
