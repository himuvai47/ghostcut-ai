from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any
from urllib import error, request

from .config import IndexerConfig
from .models import VISION_JSON_SCHEMA, VisionAnalysis, validate_vision_payload


VISION_SYSTEM_PROMPT = """You are the visual indexing component of a local video-editing system.
Analyze the ordered representative frames as evidence from ONE continuous source-video clip.
Your job is faithful visual observation for future semantic retrieval, not editing.

STRICT RULES
1. Do not choose timeline positions, narration matches, story roles, or editing decisions.
2. Describe only what the supplied frames support. Never invent identities, brands, models, locations, events, dates, or intent.
3. OBSERVATION and INFERENCE must stay separate:
   - observed_details: only directly visible facts. Never use likely, possibly, suggests, appears to, may, might, could, probably.
   - uncertain_inferences: plausible interpretations that are not directly visible. Phrase them explicitly as uncertain.
4. SUBJECT HIERARCHY matters:
   - primary_subject = dominant intentionally presented visual subject.
   - primary_product = product deliberately showcased/handled, or "none" if there is no clear product.
   - secondary_products = incidental/secondary products, including another watch worn by a presenter while a different watch is showcased.
   Never let an incidental product replace the deliberately presented product.
5. ENTITY COMPLETENESS matters:
   - objects must include concrete visible non-person entities relevant to retrieval, especially the primary/secondary products and packaging/props.
   - people must include visible people or visible person fragments when a person is materially present (for example gloved hands handling a product).
   Do not leave objects/people empty merely because the same entity is already named elsewhere.
6. Use the exact normalized enum values required by the JSON schema for shot_type, camera_motion, content_type, quality_flags and exclusion_reason.
7. SHOT TYPE is framing only. Apply these definitions consistently:
   - extreme_close_up: very small detail fills most of frame (dial text, mechanism, clasp engraving).
   - macro: macro/magnified detail with very shallow close focus.
   - close_up: product/face/hands dominate frame; little body/environment context.
   - medium_close_up: person framed roughly chest/shoulders upward.
   - medium: person framed roughly waist/torso with surrounding context.
   - medium_wide/wide: substantial body/environment visible.
   - insert: isolated detail/prop insert shot.
   - text_screen: primarily a designed text/logo/card screen.
   A product filling most of the frame is close_up even if the whole product is visible.
8. CAMERA MOTION must be conservative because evidence is sampled frames. Use "unknown" unless motion can be established from frame-to-frame evidence.
9. CONTENT TYPE should be as specific as evidence allows:
   - product_closeup: product dominates frame, no meaningful handling action.
   - product_in_box: product primarily displayed/resting in packaging or presentation box.
   - product_handling: hands/person actively rotate, adjust, remove, place, examine, or manipulate product.
   - presenter_demo: presenter/person is materially framed while demonstrating/discussing a product.
   - product_detail: product-focused footage that does not fit the more specific product categories.
   - talking_head: person speaking without product demonstration as the visual focus.
   - end_screen/intro_card/credits/logo_screen/text_screen for non-footage graphics/cards.
10. quality_flags contains ONLY actual defects from the allowed enum list. [] means no detected defect.
11. visible_text contains only text you can actually read. If none is readable, return []. Never write "No visible text" as an item.
12. usable means generally useful as source visual footage. End screens, intro cards, credits, logo screens, text-only screens, blank imagery, and severely unusable visual material should be false.
13. exclusion_reason must be "none" when usable is true. When unusable, choose the best allowed reason.
14. Do NOT output a confidence score or confidence level. Deterministic software calculates confidence from the validated evidence after your response.
15. visual_concepts may contain higher-level concepts only when strongly supported by what is visible.
16. EDITORIAL INDEX FIELDS describe the shot itself, not the narration it might later accompany:
   - brand: visible/strongly identifiable product brand, otherwise "unknown". Do not guess from style alone.
   - model_name: exact visible/strongly identifiable product model, otherwise "unknown".
   - subject_focus: what the shot visually emphasizes (for watches, distinguish dial, movement, caseback, bracelet, wrist_wear, full_product, etc.).
   - shot_angle: dominant viewing angle of the subject.
   - editorial_role: the most useful generic B-roll role supported by the image (product_beauty, technical_detail, movement_detail, wrist_lifestyle, handling_demo, retail_display, brand_identity, comparison_support, environment_context, generic_topic_broll, presenter, text_graphic, other).
   - visual_energy: static/calm/moderate/dynamic based on visible action and established camera movement.
   - lighting_style: visible lighting character only.
   - composition: broad compositional pattern only. comparison_like requires two or more products/subjects visibly arranged for comparison; do not infer comparison from narration because there is no narration.
17. Do NOT output visual_quality_score, editorial_usefulness_score, visual_family, or editorial_profile_version. Software derives those deterministically.
18. Be concise. Avoid repeating the same fact across many fields. The structured record must finish completely rather than becoming verbose.
19. Output only data matching the supplied JSON schema.
"""


class OllamaError(RuntimeError):
    pass


def _json_request(url: str, payload: dict[str, Any] | None, timeout: int) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise OllamaError(f"Cannot reach Ollama at {url}: {exc}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama returned invalid JSON envelope: {body[:500]}") from exc


def check_ollama(cfg: IndexerConfig) -> tuple[bool, str]:
    base = cfg.ollama_url.rstrip("/")
    try:
        payload = _json_request(base + "/api/tags", None, timeout=15)
    except OllamaError as exc:
        return False, str(exc)
    names = {str(m.get("name", "")) for m in payload.get("models", [])}
    if cfg.vision_model not in names:
        return False, f"Model {cfg.vision_model!r} not installed. Run: ollama pull {cfg.vision_model}"
    return True, f"Ollama reachable; model {cfg.vision_model} is installed"


def _encode_images(paths: list[Path]) -> list[str]:
    return [base64.b64encode(path.read_bytes()).decode("ascii") for path in paths]


def _select_evenly(paths: list[Path], cap: int) -> list[Path]:
    """Preserve temporal coverage when a retry must use fewer images."""
    if len(paths) <= cap:
        return list(paths)
    if cap <= 1:
        return [paths[len(paths) // 2]]
    indices = []
    for i in range(cap):
        idx = round(i * (len(paths) - 1) / (cap - 1))
        if idx not in indices:
            indices.append(idx)
    return [paths[i] for i in indices]


def _bounded_request_schema(*, ultra_compact: bool = False) -> dict[str, Any]:
    """Same semantic contract, with output-size hints that Ollama can enforce during generation."""
    schema = deepcopy(VISION_JSON_SCHEMA)
    props = schema["properties"]

    # Keep retrieval content useful while preventing runaway repetition from exhausting output tokens.
    string_caps = {
        "description": 700 if not ultra_compact else 420,
        "primary_subject": 180,
        "primary_product": 180,
        "environment": 280 if not ultra_compact else 180,
        "brand": 120,
        "model_name": 160,
    }
    for name, limit in string_caps.items():
        props[name]["maxLength"] = limit

    list_caps = {
        "secondary_subjects": 3,
        "secondary_products": 3,
        "objects": 7 if not ultra_compact else 5,
        "people": 4,
        "actions": 5 if not ultra_compact else 4,
        "visual_attributes": 7 if not ultra_compact else 5,
        "visual_concepts": 5 if not ultra_compact else 4,
        "visible_text": 8 if not ultra_compact else 6,
        "observed_details": 8 if not ultra_compact else 6,
        "uncertain_inferences": 4 if not ultra_compact else 3,
        "quality_flags": 5,
    }
    for name, limit in list_caps.items():
        if name not in props:
            continue
        props[name]["maxItems"] = limit
        item_schema = props[name].get("items")
        if isinstance(item_schema, dict) and item_schema.get("type") == "string":
            item_schema.setdefault("maxLength", 180 if not ultra_compact else 140)
    return schema


def _attempt_prompt(attempt: int) -> str:
    base = (
        "Analyze these ordered representative frames from the same continuous source clip. "
        "Create a precise retrieval-quality record. Fully populate visible entities, identify the dominant subject and "
        "deliberately presented product, keep incidental products secondary, separate direct observations from uncertainty, "
        "and use the most specific framing/content enums supported by the evidence. "
    )
    if attempt <= 1:
        return base + (
            "Keep the description to at most 3 short sentences. Keep list items short and non-redundant; "
            "prefer 3-5 useful items rather than exhaustive repetition. Identify brand/model only when supported, "
            "and classify subject_focus/editorial_role for future B-roll selection. Complete every required JSON field."
        )
    if attempt == 2:
        return base + (
            "IMPORTANT RETRY: the previous structured response did not finish cleanly. Be compact. "
            "Use 1-2 short description sentences, no more than 4 items in ordinary lists, and no repeated facts. "
            "Prioritize completing valid JSON over elaboration. Complete every required field."
        )
    return base + (
        "FINAL COMPACT RETRY: return the smallest accurate record that satisfies the schema. "
        "Use one short description sentence, short noun/verb phrases, at most 3 useful items per list where possible, "
        "and no commentary or repetition. Every required field must be present and the JSON must close completely."
    )


def _content_parse_error(content: str, envelope: dict[str, Any], exc: json.JSONDecodeError) -> OllamaError:
    reason = envelope.get("done_reason") or envelope.get("doneReason") or "unknown"
    return OllamaError(
        "Ollama assistant content was incomplete/invalid JSON "
        f"(chars={len(content)}, done_reason={reason}, line={exc.lineno}, col={exc.colno}): {exc.msg}"
    )


def analyze_clip(paths: list[Path], cfg: IndexerConfig) -> VisionAnalysis:
    if not paths:
        raise OllamaError("No representative images supplied")
    base = cfg.ollama_url.rstrip("/")
    encoded_cache: dict[tuple[str, ...], list[str]] = {}

    last_error: Exception | None = None
    attempts = max(1, cfg.model_retries)
    for attempt in range(1, attempts + 1):
        # Normal attempt uses all evidence. Only malformed/retry paths reduce image count.
        if attempt == 1:
            attempt_paths = list(paths)
        elif attempt < attempts:
            attempt_paths = _select_evenly(paths, cfg.retry_image_cap)
        else:
            attempt_paths = _select_evenly(paths, cfg.final_retry_image_cap)

        key = tuple(str(p) for p in attempt_paths)
        if key not in encoded_cache:
            encoded_cache[key] = _encode_images(attempt_paths)

        ultra_compact = attempt == attempts and attempts > 1
        payload: dict[str, Any] = {
            "model": cfg.vision_model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": _attempt_prompt(attempt), "images": encoded_cache[key]},
            ],
            "format": _bounded_request_schema(ultra_compact=ultra_compact),
            "stream": False,
            "keep_alive": cfg.ollama_keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": cfg.ollama_num_ctx,
                "num_predict": cfg.ollama_num_predict,
            },
        }

        try:
            envelope = _json_request(base + "/api/chat", payload, timeout=cfg.ollama_timeout_sec)
            content = envelope.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise OllamaError("Ollama returned an empty assistant message")
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                raise _content_parse_error(content, envelope, exc) from exc
            if not isinstance(data, dict):
                raise ValueError("Vision response root must be an object")
            return validate_vision_payload(data)
        except (OllamaError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                # Small delay only; retries are deliberately different, so don't waste time repeating the same payload.
                time.sleep(min(attempt, 2))

    raise OllamaError(f"Vision analysis failed after {attempts} attempts: {last_error}")
