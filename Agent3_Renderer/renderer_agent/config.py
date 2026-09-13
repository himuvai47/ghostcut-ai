from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib, json

@dataclass(slots=True)
class Agent3Config:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    encoder: str = "auto"
    nvenc_preset: str = "p4"
    nvenc_cq: int = 20
    x264_preset: str = "medium"
    x264_crf: int = 20
    pixel_format: str = "yuv420p"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    audio_sample_rate: int = 48000
    duration_tolerance_seconds: float = 0.10
    gap_warning_seconds: float = 0.10
    cache_enabled: bool = True

    @classmethod
    def load(cls, path: Path | None) -> "Agent3Config":
        cfg = cls()
        if path:
            data=json.loads(path.read_text(encoding="utf-8"))
            for k,v in data.items():
                if not hasattr(cfg,k):
                    raise ValueError(f"Unknown config key: {k}")
                setattr(cfg,k,v)
        if cfg.width <=0 or cfg.height<=0 or cfg.fps<=0:
            raise ValueError("width, height and fps must be positive")
        if cfg.duration_tolerance_seconds < 0 or cfg.gap_warning_seconds < 0:
            raise ValueError("duration/gap tolerances cannot be negative")
        return cfg

    def signature(self) -> str:
        raw=json.dumps(asdict(self),sort_keys=True,separators=(",",":"))
        return hashlib.sha256(raw.encode()).hexdigest()
