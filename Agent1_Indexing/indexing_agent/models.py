from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any


SHOT_TYPES = (
    "extreme_close_up", "macro", "close_up", "medium_close_up", "medium",
    "medium_wide", "wide", "overhead", "pov", "insert", "text_screen", "unknown",
)
CAMERA_MOTIONS = (
    "static", "pan", "tilt", "push_in", "pull_out", "zoom_in", "zoom_out",
    "tracking", "handheld", "orbit", "mixed", "unknown",
)
CONTENT_TYPES = (
    "product_closeup", "product_in_box", "product_handling", "presenter_demo", "product_detail",
    "talking_head", "environment", "action", "text_screen", "intro_card", "end_screen", "credits",
    "logo_screen", "other",
)
QUALITY_FLAGS = (
    "motion_blur", "out_of_focus", "underexposed", "overexposed", "obstructed",
    "mostly_black_or_blank", "severe_compression", "poor_subject_visibility",
)
EXCLUSION_REASONS = (
    "none", "end_screen", "intro_card", "credits", "logo_screen", "text_only_screen",
    "mostly_black_or_blank", "poor_visual_quality", "no_clear_subject", "other",
)
CONFIDENCE_LEVELS = ("low", "medium", "high")
SUBJECT_FOCUS = (
    "full_product", "dial", "movement", "caseback", "case_side", "bezel", "bracelet", "strap",
    "clasp", "crown_pushers", "hands_on_product", "wrist_wear", "packaging", "brand_logo",
    "retail_display", "person", "environment", "text_graphic", "multiple_products", "other", "unknown",
)
SHOT_ANGLES = (
    "front", "three_quarter", "side", "rear_caseback", "top_down", "low_angle", "high_angle",
    "wrist_perspective", "mixed", "unknown",
)
EDITORIAL_ROLES = (
    "product_beauty", "technical_detail", "movement_detail", "wrist_lifestyle", "handling_demo",
    "retail_display", "brand_identity", "comparison_support", "environment_context",
    "generic_topic_broll", "presenter", "text_graphic", "other",
)
VISUAL_ENERGIES = ("static", "calm", "moderate", "dynamic")
LIGHTING_STYLES = ("bright", "dark", "neutral", "high_contrast", "natural", "mixed", "unknown")
COMPOSITIONS = (
    "single_subject", "multi_subject", "detail_only", "person_and_product", "text_dominant",
    "environmental", "comparison_like", "unknown",
)
EDITORIAL_PROFILE_VERSION = 1

_NON_EDITORIAL_TYPES = {
    "end_screen": "end_screen",
    "intro_card": "intro_card",
    "credits": "credits",
    "logo_screen": "logo_screen",
    "text_screen": "text_only_screen",
}

_UNCERTAINTY_RE = re.compile(
    r"\b(likely|possibly|possible|perhaps|probably|suggests?|appears? to|may be|might be|could be|seems? to)\b",
    re.IGNORECASE,
)
_INTERPRETIVE_RE = re.compile(
    r"\b(review|demonstrat|presentation|promotional|advertis|professional|formal|luxury|valuable|purpose|designed to|engage|audience|retail|sale|tutorial)\b",
    re.IGNORECASE,
)
_DIRECT_OBSERVATION_RE = re.compile(
    r"^(the\s+)?(video\s+frames?|frames?|camera|watch|product|person|man|woman|hand|hands|background|box|logo|text|screen|dial|bracelet|strap|object)\b",
    re.IGNORECASE,
)
_PERSON_RE = re.compile(r"\b(person|man|woman|presenter|host|people|human|hand|hands|glove|gloved|wrist|arm)\b", re.IGNORECASE)
_PRODUCT_GENERIC_RE = re.compile(r"^(a\s+|the\s+)?(product|item|watch|wristwatch|object)$", re.IGNORECASE)
_NONE_RE = re.compile(r"^(none|unknown|n/?a|null|no clear product|no product)$", re.IGNORECASE)


@dataclass(slots=True)
class MediaInfo:
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    format_name: str = ""
    has_audio: bool = False
    rotation: int = 0


@dataclass(slots=True)
class TimeRange:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class VisionAnalysis:
    description: str
    primary_subject: str = "unknown"
    secondary_subjects: list[str] = field(default_factory=list)
    primary_product: str = "none"
    secondary_products: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    people: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    environment: str = "unknown"
    shot_type: str = "unknown"
    camera_motion: str = "unknown"
    content_type: str = "other"
    visual_attributes: list[str] = field(default_factory=list)
    visual_concepts: list[str] = field(default_factory=list)
    visible_text: list[str] = field(default_factory=list)
    observed_details: list[str] = field(default_factory=list)
    uncertain_inferences: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    usable: bool = True
    exclusion_reason: str = "none"
    confidence_level: str = "medium"
    analysis_confidence: float = 0.75
    confidence_basis: list[str] = field(default_factory=list)
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
    editorial_profile_version: int = EDITORIAL_PROFILE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Confidence is intentionally NOT requested from Qwen in v1.1.1. Software derives it
# deterministically from the validated evidence so the model cannot mechanically return
# the same high value for every clip.
VISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "primary_subject": {"type": "string"},
        "secondary_subjects": {"type": "array", "items": {"type": "string"}},
        "primary_product": {"type": "string"},
        "secondary_products": {"type": "array", "items": {"type": "string"}},
        "objects": {"type": "array", "items": {"type": "string"}},
        "people": {"type": "array", "items": {"type": "string"}},
        "actions": {"type": "array", "items": {"type": "string"}},
        "environment": {"type": "string"},
        "shot_type": {"type": "string", "enum": list(SHOT_TYPES)},
        "camera_motion": {"type": "string", "enum": list(CAMERA_MOTIONS)},
        "content_type": {"type": "string", "enum": list(CONTENT_TYPES)},
        "visual_attributes": {"type": "array", "items": {"type": "string"}},
        "visual_concepts": {"type": "array", "items": {"type": "string"}},
        "visible_text": {"type": "array", "items": {"type": "string"}},
        "observed_details": {"type": "array", "items": {"type": "string"}},
        "uncertain_inferences": {"type": "array", "items": {"type": "string"}},
        "brand": {"type": "string"},
        "model_name": {"type": "string"},
        "subject_focus": {"type": "string", "enum": list(SUBJECT_FOCUS)},
        "shot_angle": {"type": "string", "enum": list(SHOT_ANGLES)},
        "editorial_role": {"type": "string", "enum": list(EDITORIAL_ROLES)},
        "visual_energy": {"type": "string", "enum": list(VISUAL_ENERGIES)},
        "lighting_style": {"type": "string", "enum": list(LIGHTING_STYLES)},
        "composition": {"type": "string", "enum": list(COMPOSITIONS)},
        "quality_flags": {"type": "array", "items": {"type": "string", "enum": list(QUALITY_FLAGS)}},
        "usable": {"type": "boolean"},
        "exclusion_reason": {"type": "string", "enum": list(EXCLUSION_REASONS)},
    },
    "required": [
        "description", "primary_subject", "secondary_subjects", "primary_product",
        "secondary_products", "objects", "people", "actions", "environment", "shot_type",
        "camera_motion", "content_type", "visual_attributes", "visual_concepts", "visible_text",
        "observed_details", "uncertain_inferences", "brand", "model_name", "subject_focus",
        "shot_angle", "editorial_role", "visual_energy", "lighting_style", "composition",
        "quality_flags", "usable", "exclusion_reason",
    ],
    "additionalProperties": False,
}


def _str_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        item = item.strip()
        key = item.casefold()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _enum(value: Any, field_name: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of: {', '.join(allowed)}")
    return normalized


def _clean_visible_text(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        low = item.casefold().strip().rstrip(".")
        if low in {"none", "n/a", "unknown", "not visible", "no text", "no visible text"}:
            continue
        if low.startswith("no visible text") or low.startswith("no readable text"):
            continue
        out.append(item)
    return out


def _normalize_none_string(value: str, default: str) -> str:
    value = value.strip()
    if not value or _NONE_RE.match(value):
        return default
    return value


def _dedupe_extend(base: list[str], items: list[str]) -> list[str]:
    out = list(base)
    seen = {x.casefold().strip(" .") for x in out}
    for item in items:
        clean = item.strip()
        key = clean.casefold().strip(" .")
        if clean and key not in seen:
            seen.add(key)
            out.append(clean)
    return out


def _normalize_observation_and_inference(observed: list[str], inferred: list[str]) -> tuple[list[str], list[str]]:
    clean_observed: list[str] = []
    clean_inferred: list[str] = []

    # Anything explicitly hedged cannot remain an observation.
    for item in observed:
        if _UNCERTAINTY_RE.search(item):
            clean_inferred.append(item)
        else:
            clean_observed.append(item)

    # Qwen occasionally puts obvious visual facts under inference. Recover only high-confidence
    # observable statements; keep interpretive claims as inference and explicitly mark them uncertain.
    for item in inferred:
        if _UNCERTAINTY_RE.search(item):
            clean_inferred.append(item)
            continue
        if _DIRECT_OBSERVATION_RE.search(item) and not _INTERPRETIVE_RE.search(item):
            clean_observed.append(item)
            continue
        clean_inferred.append(f"Possibly: {item}" if not item.lower().startswith("possibly:") else item)

    return (
        _str_list(clean_observed, "observed_details"),
        _str_list(clean_inferred, "uncertain_inferences"),
    )


def _entity_completeness(
    primary_subject: str,
    secondary_subjects: list[str],
    primary_product: str,
    secondary_products: list[str],
    objects: list[str],
    people: list[str],
) -> tuple[list[str], list[str]]:
    products = [] if primary_product == "none" else [primary_product]
    products.extend(secondary_products)
    objects = _dedupe_extend(objects, products)

    person_candidates = [primary_subject, *secondary_subjects]
    for candidate in person_candidates:
        if candidate not in {"none", "unknown"} and _PERSON_RE.search(candidate):
            people = _dedupe_extend(people, [candidate])

    # If the dominant subject is a concrete non-person object and no product was identified,
    # preserve it in objects so structured retrieval does not lose the main visual entity.
    if primary_product == "none" and primary_subject not in {"none", "unknown"} and not _PERSON_RE.search(primary_subject):
        objects = _dedupe_extend(objects, [primary_subject])

    return objects, people


def _normalize_content_type(
    current: str,
    primary_subject: str,
    primary_product: str,
    objects: list[str],
    people: list[str],
    actions: list[str],
    description: str,
    observed: list[str],
    environment: str,
    shot_type: str,
) -> str:
    if current in _NON_EDITORIAL_TYPES or current in {"environment", "action", "talking_head"}:
        return current

    product_present = primary_product not in {"none", "unknown"}
    if not product_present:
        return current

    text = " ".join([description, environment, *observed, *objects]).casefold()
    action_text = " ".join(actions).casefold()
    observed_text = " ".join(observed).casefold()
    handling_re = r"\b(hold(?:ing|s)?|held|rotat(?:e|es|ed|ing)?|turn(?:s|ed|ing)?|adjust(?:s|ed|ing)?|manipulat(?:e|es|ed|ing)?|examin(?:e|es|ed|ing)?|remov(?:e|es|ed|ing)?)\b"
    presenter_visible = bool(people) or bool(_PERSON_RE.search(primary_subject))
    handling = bool(re.search(handling_re, action_text)) or (presenter_visible and bool(re.search(handling_re, observed_text)))
    in_box = bool(re.search(r"\b(box|case|packaging|recessed compartment|cushioned stand|display stand)\b", text))
    person_dominant = bool(re.search(r"\b(person|man|woman|presenter|host)\b", primary_subject, re.IGNORECASE))

    if presenter_visible and person_dominant and shot_type in {"medium_close_up", "medium", "medium_wide", "wide"}:
        return "presenter_demo"
    if handling:
        return "product_handling"
    if in_box and not presenter_visible:
        return "product_in_box"
    if shot_type in {"extreme_close_up", "macro", "close_up", "insert"} and not person_dominant:
        return "product_closeup"
    if in_box:
        return "product_in_box"
    return "product_detail"


def _normalize_editorial_role(
    current: str, *, content_type: str, subject_focus: str, composition: str, people: list[str]
) -> str:
    # Deterministic corrections for obvious visual cases keep filler-selection metadata consistent.
    if content_type in {"end_screen", "intro_card", "credits", "logo_screen", "text_screen"}:
        return "text_graphic"
    if composition == "comparison_like":
        return "comparison_support"
    if subject_focus == "movement":
        return "movement_detail"
    if subject_focus == "wrist_wear":
        return "wrist_lifestyle"
    if subject_focus == "retail_display":
        return "retail_display"
    if subject_focus == "brand_logo":
        return "brand_identity"
    if content_type == "product_handling":
        return "handling_demo"
    if content_type in {"presenter_demo", "talking_head"}:
        return "presenter"
    if content_type == "environment" and current in {"other", "generic_topic_broll"}:
        return "environment_context"
    if subject_focus in {"dial", "caseback", "case_side", "bezel", "bracelet", "strap", "clasp", "crown_pushers"}:
        if current in {"other", "generic_topic_broll", "product_beauty"}:
            return "technical_detail"
    if content_type in {"product_closeup", "product_in_box", "product_detail"} and current == "other":
        return "product_beauty"
    return current


def _normalize_shot_type(shot_type: str, content_type: str, primary_subject: str, people: list[str]) -> str:
    # Obvious text cards are not medium shots simply because the model saw the whole screen.
    if content_type in {"end_screen", "intro_card", "credits", "logo_screen", "text_screen"}:
        return "text_screen"

    # Product-only beauty/box shots are frequently mislabeled as medium by small VL models.
    # Correct only the conservative case where a person is not a dominant visible subject.
    person_dominant = bool(re.search(r"\b(person|man|woman|presenter|host)\b", primary_subject, re.IGNORECASE))
    if content_type in {"product_closeup", "product_in_box"} and shot_type in {"medium", "medium_wide"} and not person_dominant and not people:
        return "close_up"
    return shot_type


def _derive_confidence(
    *, description: str, primary_subject: str, primary_product: str, content_type: str,
    shot_type: str, observed: list[str], inferred: list[str], visible_text: list[str],
    quality_flags: list[str], usable: bool,
) -> tuple[str, float, list[str]]:
    score = 0.82
    basis: list[str] = []

    if primary_subject in {"unknown", "none"}:
        score -= 0.22
        basis.append("primary subject unclear")
    else:
        basis.append("primary subject identified")

    product_related = content_type.startswith("product_") or content_type == "presenter_demo"
    if product_related and primary_product in {"none", "unknown"}:
        score -= 0.12
        basis.append("product identity missing")
    elif primary_product not in {"none", "unknown"}:
        if _PRODUCT_GENERIC_RE.match(primary_product.strip()):
            score -= 0.04
            basis.append("product identified only generically")
        else:
            score += 0.03
            basis.append("product has specific visual identity")

    if len(observed) >= 4:
        score += 0.04
        basis.append("multiple direct observations agree")
    elif not observed:
        score -= 0.08
        basis.append("few direct observations")

    if visible_text:
        score += 0.02
        basis.append("readable visual text supports identification")

    if shot_type == "unknown":
        score -= 0.04
        basis.append("framing uncertain")

    if inferred:
        penalty = min(0.10, 0.02 * len(inferred))
        score -= penalty
        basis.append(f"{len(inferred)} uncertain inference(s)")

    if quality_flags:
        penalty = min(0.18, 0.06 * len(quality_flags))
        score -= penalty
        basis.append("visual quality limitation detected")

    if len(description.strip()) < 60:
        score -= 0.04
        basis.append("description is sparse")

    if not usable and content_type in _NON_EDITORIAL_TYPES:
        # Classification of a clear end card/credits screen can still be highly reliable.
        score = max(score, 0.88)
        basis.append("non-editorial screen classification is visually explicit")

    score = max(0.35, min(0.95, score))
    score = round(score, 2)
    if score >= 0.84:
        level = "high"
    elif score >= 0.64:
        level = "medium"
    else:
        level = "low"
    return level, score, basis



def _slug(value: str, fallback: str = "unknown") -> str:
    value = value.strip().casefold()
    if not value or value in {"none", "unknown", "n/a", "null"}:
        return fallback
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value[:80] or fallback


def _derive_editorial_scores(
    *, usable: bool, quality_flags: list[str], analysis_confidence: float, shot_type: str,
    content_type: str, editorial_role: str, subject_focus: str, brand: str, model_name: str,
) -> tuple[float, float]:
    quality = 0.90
    penalties = {
        "motion_blur": 0.16, "out_of_focus": 0.22, "underexposed": 0.10,
        "overexposed": 0.10, "obstructed": 0.12, "mostly_black_or_blank": 0.55,
        "severe_compression": 0.18, "poor_subject_visibility": 0.24,
    }
    for flag in quality_flags:
        quality -= penalties.get(flag, 0.05)
    if shot_type == "unknown":
        quality -= 0.04
    quality = 0.70 * quality + 0.30 * analysis_confidence
    if not usable:
        quality = min(quality, 0.20)
    quality = round(max(0.0, min(1.0, quality)), 2)

    if not usable:
        return quality, 0.0
    usefulness = 0.42 + 0.34 * quality + 0.16 * analysis_confidence
    role_boost = {
        "product_beauty": 0.10, "technical_detail": 0.09, "movement_detail": 0.10,
        "wrist_lifestyle": 0.08, "handling_demo": 0.08, "retail_display": 0.05,
        "brand_identity": 0.04, "comparison_support": 0.07, "environment_context": 0.03,
        "generic_topic_broll": 0.01, "presenter": 0.02, "text_graphic": -0.04, "other": 0.0,
    }
    usefulness += role_boost.get(editorial_role, 0.0)
    if subject_focus == "unknown":
        usefulness -= 0.06
    if brand != "unknown":
        usefulness += 0.03
    if model_name != "unknown":
        usefulness += 0.03
    if content_type in {"end_screen", "intro_card", "credits", "logo_screen", "text_screen"}:
        usefulness = min(usefulness, 0.15)
    return quality, round(max(0.0, min(1.0, usefulness)), 2)


def _derive_visual_family(
    *, brand: str, model_name: str, primary_product: str, editorial_role: str,
    subject_focus: str, shot_type: str, shot_angle: str, lighting_style: str,
    camera_motion: str, composition: str, content_type: str,
) -> str:
    if brand != "unknown" and model_name != "unknown":
        identity = f"{brand} {model_name}"
    else:
        identity = model_name if model_name != "unknown" else brand
    if identity == "unknown" and primary_product not in {"none", "unknown"}:
        identity = primary_product
    identity_slug = _slug(identity, _slug(content_type, "visual"))
    parts = [identity_slug, editorial_role, subject_focus, shot_type, shot_angle, lighting_style]
    if camera_motion not in {"unknown", "mixed"}:
        parts.append(camera_motion)
    if composition in {"comparison_like", "person_and_product", "environmental"}:
        parts.append(composition)
    return "__".join(_slug(str(x)) for x in parts if x)


def validate_vision_payload(data: dict[str, Any]) -> VisionAnalysis:
    required = set(VISION_JSON_SCHEMA["required"])
    missing = required - set(data)
    if missing:
        raise ValueError(f"Vision JSON missing fields: {', '.join(sorted(missing))}")
    unknown = set(data) - set(VISION_JSON_SCHEMA["properties"])
    if unknown:
        raise ValueError(f"Vision JSON has unexpected fields: {', '.join(sorted(unknown))}")

    description = data["description"]
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")

    primary_subject = data["primary_subject"]
    primary_product = data["primary_product"]
    environment = data["environment"]
    for name, value in (("primary_subject", primary_subject), ("primary_product", primary_product), ("environment", environment)):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")

    primary_subject = _normalize_none_string(primary_subject, "unknown")
    primary_product = _normalize_none_string(primary_product, "none")
    environment = _normalize_none_string(environment, "unknown")
    brand = _normalize_none_string(str(data["brand"]), "unknown")
    model_name = _normalize_none_string(str(data["model_name"]), "unknown")

    shot_type = _enum(data["shot_type"], "shot_type", SHOT_TYPES)
    camera_motion = _enum(data["camera_motion"], "camera_motion", CAMERA_MOTIONS)
    content_type = _enum(data["content_type"], "content_type", CONTENT_TYPES)
    exclusion_reason = _enum(data["exclusion_reason"], "exclusion_reason", EXCLUSION_REASONS)
    subject_focus = _enum(data["subject_focus"], "subject_focus", SUBJECT_FOCUS)
    shot_angle = _enum(data["shot_angle"], "shot_angle", SHOT_ANGLES)
    editorial_role = _enum(data["editorial_role"], "editorial_role", EDITORIAL_ROLES)
    visual_energy = _enum(data["visual_energy"], "visual_energy", VISUAL_ENERGIES)
    lighting_style = _enum(data["lighting_style"], "lighting_style", LIGHTING_STYLES)
    composition = _enum(data["composition"], "composition", COMPOSITIONS)

    usable = data["usable"]
    if not isinstance(usable, bool):
        raise ValueError("usable must be boolean")

    quality_flags = _str_list(data["quality_flags"], "quality_flags")
    for flag in quality_flags:
        if flag not in QUALITY_FLAGS:
            raise ValueError(f"Invalid quality flag: {flag}")

    secondary_subjects = _str_list(data["secondary_subjects"], "secondary_subjects")
    secondary_products = _str_list(data["secondary_products"], "secondary_products")
    objects = _str_list(data["objects"], "objects")
    people = _str_list(data["people"], "people")
    actions = _str_list(data["actions"], "actions")
    visual_attributes = _str_list(data["visual_attributes"], "visual_attributes")
    visual_concepts = _str_list(data["visual_concepts"], "visual_concepts")
    visible_text = _clean_visible_text(_str_list(data["visible_text"], "visible_text"))
    observed = _str_list(data["observed_details"], "observed_details")
    inferred = _str_list(data["uncertain_inferences"], "uncertain_inferences")
    observed, inferred = _normalize_observation_and_inference(observed, inferred)

    objects, people = _entity_completeness(
        primary_subject, secondary_subjects, primary_product, secondary_products, objects, people
    )

    content_type = _normalize_content_type(
        content_type, primary_subject, primary_product, objects, people, actions,
        description.strip(), observed, environment, shot_type,
    )
    shot_type = _normalize_shot_type(shot_type, content_type, primary_subject, people)
    editorial_role = _normalize_editorial_role(
        editorial_role, content_type=content_type, subject_focus=subject_focus,
        composition=composition, people=people,
    )

    # Deterministic usability policy. Qwen identifies the content class; software enforces
    # that source outros/cards/credits do not enter normal future footage retrieval.
    if content_type in _NON_EDITORIAL_TYPES:
        usable = False
        exclusion_reason = _NON_EDITORIAL_TYPES[content_type]
    if "mostly_black_or_blank" in quality_flags:
        usable = False
        exclusion_reason = "mostly_black_or_blank"
    if exclusion_reason != "none":
        usable = False
    elif not usable:
        exclusion_reason = "other"

    confidence_level, analysis_confidence, confidence_basis = _derive_confidence(
        description=description.strip(), primary_subject=primary_subject, primary_product=primary_product,
        content_type=content_type, shot_type=shot_type, observed=observed, inferred=inferred,
        visible_text=visible_text, quality_flags=quality_flags, usable=usable,
    )
    visual_quality_score, editorial_usefulness_score = _derive_editorial_scores(
        usable=usable, quality_flags=quality_flags, analysis_confidence=analysis_confidence,
        shot_type=shot_type, content_type=content_type, editorial_role=editorial_role,
        subject_focus=subject_focus, brand=brand, model_name=model_name,
    )
    visual_family = _derive_visual_family(
        brand=brand, model_name=model_name, primary_product=primary_product, editorial_role=editorial_role,
        subject_focus=subject_focus, shot_type=shot_type, shot_angle=shot_angle,
        lighting_style=lighting_style, camera_motion=camera_motion, composition=composition,
        content_type=content_type,
    )

    return VisionAnalysis(
        description=description.strip(),
        primary_subject=primary_subject,
        secondary_subjects=secondary_subjects,
        primary_product=primary_product,
        secondary_products=secondary_products,
        objects=objects,
        people=people,
        actions=actions,
        environment=environment,
        shot_type=shot_type,
        camera_motion=camera_motion,
        content_type=content_type,
        visual_attributes=visual_attributes,
        visual_concepts=visual_concepts,
        visible_text=visible_text,
        observed_details=observed,
        uncertain_inferences=inferred,
        quality_flags=quality_flags,
        usable=usable,
        exclusion_reason=exclusion_reason,
        confidence_level=confidence_level,
        analysis_confidence=analysis_confidence,
        confidence_basis=confidence_basis,
        brand=brand,
        model_name=model_name,
        subject_focus=subject_focus,
        shot_angle=shot_angle,
        editorial_role=editorial_role,
        visual_energy=visual_energy,
        lighting_style=lighting_style,
        composition=composition,
        visual_family=visual_family,
        visual_quality_score=visual_quality_score,
        editorial_usefulness_score=editorial_usefulness_score,
        editorial_profile_version=EDITORIAL_PROFILE_VERSION,
    )
