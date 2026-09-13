from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
from pathlib import Path

@dataclass(slots=True)
class Agent2Config:
    ollama_url: str = "http://127.0.0.1:11434"
    reasoning_model: str = "qwen3.5:9b"
    embedding_model: str = "nomic-embed-text"
    face_model: str = "qwen2.5vl:7b"  # deprecated in v1.0.5; kept for old config files
    whisper_model: str = "small.en"
    whisper_language: str = "en"
    whisper_cli: str = "tools/whisper/bin/whisper-cli.exe"
    whisper_model_path: str = "tools/whisper/models/ggml-small.en.bin"
    retrieval_top_k: int = 30
    rerank_top_k: int = 8
    min_unused_retrieval_score: float = 0.25
    min_repeat_retrieval_score: float = 0.34
    max_clip_uses: int = 2
    min_repeat_segment_gap: int = 2
    min_visual_clip_seconds: float = 3.0
    max_visual_clip_seconds: float = 8.0
    prefer_longer_clips: bool = True
    target_visual_segment_seconds: float = 8.0
    max_visual_segment_seconds: float = 10.0
    broad_fallback_min_retrieval_score: float = 0.16
    min_unused_clips_before_repeat: int = 12
    diversity_source_cooldown_segments: int = 2
    diversity_same_source_near_seconds: float = 20.0
    diversity_semantic_duplicate_threshold: float = 0.975
    diversity_min_alternatives: int = 6
    diversity_source_balance_penalty: float = 0.025
    editorial_freedom_enabled: bool = True
    brand_filler_min_retrieval_score: float = 0.14
    strict_face_policy: bool = True  # deprecated compatibility field
    allow_uncertain_face_status: bool = True
    face_screen_all_candidates: bool = True  # deprecated compatibility field
    max_planner_words: int = 1600
    global_review_enabled: bool = True
    max_global_rematches: int = 1
    qwen_num_predict: int = 4096
    qwen_temperature: float = 0.0
    face_num_predict: int = 256
    face_scan_interval_seconds: float = 0.75
    face_scan_max_frames: int = 16
    face_batch_size: int = 4
    face_frame_width: int = 768

    @classmethod
    def load(cls, path: Path | None) -> "Agent2Config":
        cfg = cls()
        if path:
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if not hasattr(cfg, k):
                    raise ValueError(f"Unknown config key: {k}")
                setattr(cfg, k, v)
        return cfg

    def signature(self) -> str:
        stable = asdict(self).copy()
        stable.pop("whisper_cli", None)
        stable.pop("whisper_model_path", None)
        # Agent 2 no longer performs vision/face inference; these legacy settings must not invalidate matching.
        for k in ("face_model","strict_face_policy","face_screen_all_candidates","face_num_predict","face_scan_interval_seconds","face_scan_max_frames","face_batch_size","face_frame_width"):
            stable.pop(k, None)
        raw = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()
