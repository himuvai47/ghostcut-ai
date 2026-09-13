from __future__ import annotations

from .models import CandidateJudgment, MATCH_LEVELS
from .ollama import chat_fast

# v1.0.4: keep the model output deliberately small.  The first real Agent-2
# run showed that asking Qwen3.5 to emit a verbose judgment object for 8
# candidates can exhaust the generation/context budget and truncate JSON.
RERANK_VERSION = "rerank-v3-fast-relevance"
MATCH_VALUES = ["strong", "partial", "none", "conflict"]
SCHEMA = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clip_id": {"type": "string"},
                    "subject_match": {"type": "string", "enum": MATCH_VALUES},
                    "attribute_match": {"type": "string", "enum": MATCH_VALUES},
                    "hard_requirement_failed": {"type": "boolean"},
                    "decision": {"type": "string", "enum": list(MATCH_LEVELS)},
                    "reason": {"type": "string"},
                },
                "required": [
                    "clip_id",
                    "subject_match",
                    "attribute_match",
                    "hard_requirement_failed",
                    "decision",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}


def _short(text, limit=260):
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def _small_list(values, n=4, each=90):
    return [_short(v, each) for v in list(values or [])[:n]]


def _requirement_payload(req):
    return {
        "narration": _short(req.narration_text, 260),
        "requested_visual": _short(req.requested_visual, 220),
        "primary_subject": _short(req.primary_subject, 120),
        "required_entities": _small_list(req.required_entities, 4, 60),
        "soft_attributes": _small_list(req.required_attributes, 4, 70),
        "specificity": req.specificity,
        "match_mode": getattr(req, "match_mode", "thematic"),
        "context_anchor": _short(getattr(req, "context_anchor", ""), 100),
    }


def _candidate_payload(c):
    clip = c.clip
    return {
        "clip_id": clip.clip_id,
        "description": _short(clip.description, 260),
        "primary_subject": _short(clip.primary_subject, 100),
        "primary_product": _short(clip.primary_product, 100),
        "secondary_products": _small_list(clip.secondary_products, 2, 70),
        "shot_type": clip.shot_type,
        "content_type": clip.content_type,
        "attributes": _small_list(clip.visual_attributes, 5, 70),
        "concepts": _small_list(clip.visual_concepts, 4, 70),
        "visible_text": _small_list(clip.visible_text, 4, 50),
        "retrieval_score": round(float(c.retrieval_score), 3),
        "face_status": c.face_status,
    }


def _to_judgment(item):
    # The detailed framing/concept fields are not used for selection; keeping them
    # as neutral/derived values preserves the existing public dataclass contract
    # while making the model response far smaller and more reliable.
    attr = item["attribute_match"]
    return CandidateJudgment(
        clip_id=item["clip_id"],
        subject_match=item["subject_match"],
        attribute_match=attr,
        framing_match="partial",
        concept_match=attr,
        hard_requirement_failed=bool(item["hard_requirement_failed"]),
        conflicts=[],
        decision=item["decision"],
        reason=_short(item["reason"], 220),
    )


def _reject_for_runtime(candidate, exc):
    # Never crash the whole edit because one candidate judgment was malformed.
    # A failed judgment is conservatively rejected, never silently accepted.
    detail = _short(f"Local reranker could not validate this candidate: {type(exc).__name__}: {exc}", 200)
    return CandidateJudgment(
        clip_id=candidate.clip.clip_id,
        subject_match="none",
        attribute_match="none",
        framing_match="partial",
        concept_match="none",
        hard_requirement_failed=False,
        conflicts=[],
        decision="reject",
        reason=detail,
    )


def rerank(cfg, req, candidates):
    """Judge a small candidate batch with bounded output and recursive fallback.

    If a multi-candidate response is malformed/truncated, split the batch and retry
    smaller groups. If even one-candidate structured output cannot be validated,
    reject that candidate rather than aborting the entire Agent-2 run.
    """
    candidates = list(candidates)
    if not candidates:
        return []
    if len(candidates) > 2:
        out = []
        for i in range(0, len(candidates), 2):
            out.extend(rerank(cfg, req, candidates[i : i + 2]))
        return out

    allowed = {c.clip.clip_id for c in candidates}
    payload = [_candidate_payload(c) for c in candidates]
    prompt = (
        "Rank indexed footage for ONE faceless B-roll narration unit. Judge only supplied clips. "
        "This system values GENERAL VISUAL RELEVANCE, not perfect literal coverage. If the main subject/product is correct and the clip is broadly related, it can be acceptable even when fine narration details are not visibly proven. "
        "Use match_mode as the editorial contract: literal means exact named product/detail matters when supplied; thematic means broad topic/brand relevance is enough and abstract narration details do NOT need to be visible; brand_filler means attractive same-brand/same-topic B-roll is intentionally correct even when it does not illustrate the sentence literally. "
        "soft_attributes such as color, hour markers, texture, finishing, hand shape, exact component detail, framing, or camera motion are preferences ONLY. Missing them must NEVER cause hard_requirement_failed by itself. "
        "A hard failure is reserved for a clearly wrong required named product/entity or a clip whose main subject is plainly unrelated to the narration. "
        "Preferred shot/content type is only a preference unless narration explicitly requires it. Do not require pan/zoom/animation/montage/effects unless narration itself says so. "
        "Face policy: face_visible clips should never be accepted; uncertain may be used. "
        "Prefer strong/acceptable decisions generously for related B-roll. Return exactly one compact judgment per clip. Keep each reason to ONE short sentence, ideally under 16 words.\n"
        f"REQUIREMENT={_requirement_payload(req)}\nCANDIDATES={payload}"
    )
    try:
        data = chat_fast(
            cfg.ollama_url,
            cfg.reasoning_model,
            [{"role": "user", "content": prompt}],
            SCHEMA,
            min(int(cfg.qwen_num_predict), 768),
            cfg.qwen_temperature,
        )
        by_id = {}
        for item in data.get("judgments", []):
            cid = item.get("clip_id")
            if cid in allowed and cid not in by_id:
                by_id[cid] = _to_judgment(item)
        # Missing judgments are conservative rejects; never fabricate approval.
        return [by_id.get(c.clip.clip_id) or _reject_for_runtime(c, ValueError("candidate omitted from model response")) for c in candidates]
    except Exception as exc:
        if len(candidates) > 1:
            mid = max(1, len(candidates) // 2)
            return rerank(cfg, req, candidates[:mid]) + rerank(cfg, req, candidates[mid:])
        return [_reject_for_runtime(candidates[0], exc)]
