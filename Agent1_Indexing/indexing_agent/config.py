from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from . import PIPELINE_VERSION, PROMPT_VERSION, SCHEMA_VERSION


@dataclass(slots=True)
class IndexerConfig:
    # Models / local runtime
    ollama_url: str = "http://127.0.0.1:11434"
    vision_model: str = "qwen2.5vl:7b"
    ollama_timeout_sec: int = 600
    ollama_keep_alive: str = "5m"
    ollama_num_ctx: int = 8192
    # Explicit generation budget prevents small/local Ollama defaults from truncating structured JSON.
    # Transport-only setting: intentionally excluded from index_signature so v1.1.2 can resume v1.1.1 caches.
    ollama_num_predict: int = 3072
    model_retries: int = 3
    # On malformed/truncated JSON, later attempts progressively reduce image count to free context.
    retry_image_cap: int = 3
    final_retry_image_cap: int = 2

    # Supported media
    supported_extensions: tuple[str, ...] = (
        ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts"
    )

    # Fingerprinting
    fingerprint_mode: str = "sampled"  # sampled | full
    fingerprint_sample_bytes: int = 4 * 1024 * 1024

    # Physical shot detection
    scene_threshold: float = 0.32
    min_shot_sec: float = 0.60

    # Semantic clip subdivision (v1.1: conservative, peak-based and adaptive)
    min_semantic_clip_sec: float = 2.25
    max_semantic_clip_sec: float = 12.0
    semantic_probe_interval_sec: float = 1.25
    semantic_diff_threshold: float = 0.13
    semantic_adaptive_margin: float = 0.045
    semantic_min_boundary_spacing_sec: float = 4.0
    semantic_merge_diff_threshold: float = 0.075
    semantic_extra_split_allowance: int = 2

    # Representative frame sampling
    frame_edge_pad_sec: float = 0.15
    frame_max_width: int = 1280
    short_clip_frame_count: int = 2
    normal_clip_frame_count: int = 3
    long_clip_frame_count: int = 4
    very_long_clip_frame_count: int = 5

    # Quality pre-check
    skip_nearly_black_clips: bool = True
    black_mean_threshold: float = 0.015
    black_std_threshold: float = 0.020

    # Pipeline identity
    schema_version: int = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    prompt_version: str = PROMPT_VERSION

    @classmethod
    def load(cls, path: str | Path | None) -> "IndexerConfig":
        if path is None:
            return cls()
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        allowed = {f.name for f in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        if "supported_extensions" in raw:
            raw["supported_extensions"] = tuple(raw["supported_extensions"])
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not 0.0 < self.scene_threshold < 1.0:
            raise ValueError("scene_threshold must be between 0 and 1")
        if self.min_shot_sec <= 0:
            raise ValueError("min_shot_sec must be > 0")
        if self.min_semantic_clip_sec <= 0:
            raise ValueError("min_semantic_clip_sec must be > 0")
        if self.max_semantic_clip_sec < self.min_semantic_clip_sec:
            raise ValueError("max_semantic_clip_sec must be >= min_semantic_clip_sec")
        if self.semantic_probe_interval_sec <= 0:
            raise ValueError("semantic_probe_interval_sec must be > 0")
        if not 0.0 <= self.semantic_diff_threshold <= 1.0:
            raise ValueError("semantic_diff_threshold must be in [0,1]")
        if not 0.0 <= self.semantic_adaptive_margin <= 1.0:
            raise ValueError("semantic_adaptive_margin must be in [0,1]")
        if self.semantic_min_boundary_spacing_sec < self.min_semantic_clip_sec:
            raise ValueError("semantic_min_boundary_spacing_sec must be >= min_semantic_clip_sec")
        if not 0.0 <= self.semantic_merge_diff_threshold <= 1.0:
            raise ValueError("semantic_merge_diff_threshold must be in [0,1]")
        if self.semantic_extra_split_allowance < 0:
            raise ValueError("semantic_extra_split_allowance must be >= 0")
        if self.frame_max_width < 256:
            raise ValueError("frame_max_width must be >= 256")
        if self.ollama_num_ctx < 2048:
            raise ValueError("ollama_num_ctx must be >= 2048")
        if self.ollama_num_predict < 512:
            raise ValueError("ollama_num_predict must be >= 512")
        if self.model_retries < 1:
            raise ValueError("model_retries must be >= 1")
        if self.retry_image_cap < 1 or self.final_retry_image_cap < 1:
            raise ValueError("retry image caps must be >= 1")
        if self.fingerprint_mode not in {"sampled", "full"}:
            raise ValueError("fingerprint_mode must be sampled or full")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["supported_extensions"] = list(self.supported_extensions)
        return result

    def _signature_values(self, *, pipeline_version: str, prompt_version: str, schema_version: int) -> dict[str, Any]:
        return {
            "scene_threshold": self.scene_threshold,
            "min_shot_sec": self.min_shot_sec,
            "min_semantic_clip_sec": self.min_semantic_clip_sec,
            "max_semantic_clip_sec": self.max_semantic_clip_sec,
            "semantic_probe_interval_sec": self.semantic_probe_interval_sec,
            "semantic_diff_threshold": self.semantic_diff_threshold,
            "semantic_adaptive_margin": self.semantic_adaptive_margin,
            "semantic_min_boundary_spacing_sec": self.semantic_min_boundary_spacing_sec,
            "semantic_merge_diff_threshold": self.semantic_merge_diff_threshold,
            "semantic_extra_split_allowance": self.semantic_extra_split_allowance,
            "frame_edge_pad_sec": self.frame_edge_pad_sec,
            "frame_max_width": self.frame_max_width,
            "short_clip_frame_count": self.short_clip_frame_count,
            "normal_clip_frame_count": self.normal_clip_frame_count,
            "long_clip_frame_count": self.long_clip_frame_count,
            "very_long_clip_frame_count": self.very_long_clip_frame_count,
            "skip_nearly_black_clips": self.skip_nearly_black_clips,
            "black_mean_threshold": self.black_mean_threshold,
            "black_std_threshold": self.black_std_threshold,
            "vision_model": self.vision_model,
            "schema_version": schema_version,
            "pipeline_version": pipeline_version,
            "prompt_version": prompt_version,
        }

    @staticmethod
    def _hash_signature(values: dict[str, Any]) -> str:
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def index_signature(self) -> str:
        # Any setting that can alter segmentation, image evidence, model output,
        # or schema should invalidate prior clip analyses.
        relevant = self._signature_values(
            pipeline_version=self.pipeline_version,
            prompt_version=self.prompt_version,
            schema_version=self.schema_version,
        )
        return self._hash_signature(relevant)

    def legacy_v121_index_signature(self) -> str:
        """Signature of v1.2.1 using the current user's segmentation/model settings.

        This lets v1.3 safely recognize a v1.2.1 index as structurally compatible and
        upgrade only the vision analysis instead of destroying/re-segmenting the library.
        """
        relevant = self._signature_values(
            pipeline_version="agent1-v1.1.1",
            prompt_version="vision-v3",
            schema_version=3,
        )
        return self._hash_signature(relevant)

    def segmentation_signature(self) -> str:
        """Identity of deterministic segmentation/frame evidence only."""
        relevant = self._signature_values(
            pipeline_version="segmentation-only",
            prompt_version="not-applicable",
            schema_version=0,
        )
        for key in ("vision_model", "pipeline_version", "prompt_version", "schema_version"):
            relevant.pop(key, None)
        return self._hash_signature(relevant)
