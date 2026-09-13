from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

VISUAL_INTENTS = (
    "literal_product", "literal_product_detail", "person_or_action", "process",
    "environment", "historical_context", "conceptual", "comparison", "presenter_context",
)
SPECIFICITY = ("strict", "high", "normal", "conceptual")
MATCH_LEVELS = ("strong_match", "acceptable_match", "weak_match", "reject")
MATCH_MODES = ("literal", "thematic", "brand_filler")

@dataclass(slots=True)
class TimedWord:
    id: str
    text: str
    start: float
    end: float
    aligned: bool = True
    asr_text: str | None = None

@dataclass(slots=True)
class VisualRequirement:
    segment_id: str
    start_word_id: str
    end_word_id: str
    narration_text: str
    visual_intent: str
    requested_visual: str
    primary_subject: str
    required_entities: list[str] = field(default_factory=list)
    required_attributes: list[str] = field(default_factory=list)
    preferred_shot_types: list[str] = field(default_factory=list)
    preferred_content_types: list[str] = field(default_factory=list)
    visual_concepts: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    specificity: str = "normal"
    match_mode: str = "thematic"
    context_anchor: str = ""
    speech_start: float = 0.0
    speech_end: float = 0.0
    timeline_start: float = 0.0
    timeline_end: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.timeline_end - self.timeline_start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class IndexedClip:
    clip_id: str
    source_id: str
    source_video: str
    source_path: str
    start_time: float
    end_time: float
    duration: float
    description: str
    primary_subject: str
    primary_product: str
    secondary_products: list[str]
    objects: list[str]
    people: list[str]
    actions: list[str]
    environment: str
    shot_type: str
    camera_motion: str
    content_type: str
    visual_attributes: list[str]
    visual_concepts: list[str]
    visible_text: list[str]
    observed_details: list[str]
    uncertain_inferences: list[str]
    usable: bool
    analysis_confidence: float
    representative_paths: list[str]
    representative_times: list[float]
    analysis_hash: str
    face_status: str = "unknown"
    face_scan_version: int | None = None
    face_scan_mode: str = ""
    # Agent 1 v1.3 Editorial Vision metadata. Defaults preserve compatibility
    # with legacy v1.2.x indexes and older tests/constructors.
    brand: str = "unknown"
    model_name: str = "unknown"
    subject_focus: str = "unknown"
    shot_angle: str = "unknown"
    editorial_role: str = "other"
    visual_energy: str = "calm"
    lighting_style: str = "unknown"
    composition: str = "unknown"
    visual_family: str = "unknown"
    visual_quality_score: float = 0.5
    editorial_usefulness_score: float = 0.5
    editorial_profile_version: int = 0

    @property
    def has_editorial_profile(self) -> bool:
        return int(self.editorial_profile_version or 0) >= 1

@dataclass(slots=True)
class Candidate:
    clip: IndexedClip
    semantic_score: float
    lexical_score: float
    structured_score: float
    agent1_score: float
    retrieval_score: float
    face_status: str = "unchecked"
    face_reason: str = ""
    usage_count: int = 0
    last_used_segment: int | None = None
    diversity_penalty: float = 0.0
    diversity_reasons: list[str] = field(default_factory=list)
    near_duplicate: bool = False
    editorial_bonus: float = 0.0
    editorial_reasons: list[str] = field(default_factory=list)

    def compact_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip.clip_id,
            "source_video": self.clip.source_video,
            "source_start": round(self.clip.start_time, 3),
            "source_end": round(self.clip.end_time, 3),
            "duration": round(self.clip.duration, 3),
            "description": self.clip.description,
            "primary_subject": self.clip.primary_subject,
            "primary_product": self.clip.primary_product,
            "secondary_products": self.clip.secondary_products,
            "shot_type": self.clip.shot_type,
            "content_type": self.clip.content_type,
            "visual_attributes": self.clip.visual_attributes,
            "visual_concepts": self.clip.visual_concepts,
            "visible_text": self.clip.visible_text,
            "analysis_confidence": self.clip.analysis_confidence,
            "retrieval_score": round(self.retrieval_score, 4),
            "usage_count": self.usage_count,
            "diversity_penalty": round(self.diversity_penalty, 4),
            "diversity_reasons": self.diversity_reasons,
            "near_duplicate": self.near_duplicate,
            "editorial_bonus": round(self.editorial_bonus, 4),
            "editorial_reasons": self.editorial_reasons,
            "agent1_editorial_profile_version": self.clip.editorial_profile_version,
            "brand": self.clip.brand,
            "model_name": self.clip.model_name,
            "subject_focus": self.clip.subject_focus,
            "shot_angle": self.clip.shot_angle,
            "editorial_role": self.clip.editorial_role,
            "visual_family": self.clip.visual_family,
            "visual_quality_score": round(float(self.clip.visual_quality_score), 3),
            "editorial_usefulness_score": round(float(self.clip.editorial_usefulness_score), 3),
        }

@dataclass(slots=True)
class CandidateJudgment:
    clip_id: str
    subject_match: str
    attribute_match: str
    framing_match: str
    concept_match: str
    hard_requirement_failed: bool
    conflicts: list[str]
    decision: str
    reason: str

@dataclass(slots=True)
class TimelineClip:
    clip_id: str
    source: str
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float
    reason: str
    match_level: str
    confidence: float
    use_number: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
