from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any

from .db import IndexDB
from .models import (
    CAMERA_MOTIONS, COMPOSITIONS, CONFIDENCE_LEVELS, CONTENT_TYPES, EDITORIAL_ROLES,
    EXCLUSION_REASONS, LIGHTING_STYLES, QUALITY_FLAGS, SHOT_ANGLES, SHOT_TYPES,
    SUBJECT_FOCUS, VISUAL_ENERGIES,
)


_UNCERTAINTY_RE = re.compile(
    r"\b(likely|possibly|possible|perhaps|probably|suggests?|appears? to|may be|might be|could be|seems? to)\b",
    re.IGNORECASE,
)
_PERSON_RE = re.compile(r"\b(person|man|woman|presenter|host|people|human|hand|hands|glove|gloved|wrist|arm)\b", re.IGNORECASE)


def _close(a: float, b: float, tol: float = 0.08) -> bool:
    return abs(a - b) <= tol


def _tokens(analysis: dict[str, Any]) -> set[str]:
    parts: list[str] = []
    for key in ("primary_subject", "primary_product", "shot_type", "content_type"):
        val = analysis.get(key)
        if isinstance(val, str):
            parts.append(val)
    for key in ("secondary_subjects", "secondary_products", "objects", "visual_attributes", "visual_concepts"):
        val = analysis.get(key)
        if isinstance(val, list):
            parts.extend(str(x) for x in val)
    text = " ".join(parts).lower().replace("_", " ")
    return {t for t in re.findall(r"[a-z0-9]+", text) if len(t) > 2 and t not in {"with", "from", "unknown", "none"}}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _contains_entity(items: list[Any], entity: str) -> bool:
    target = entity.casefold().strip(" .")
    if not target or target in {"none", "unknown"}:
        return True
    for item in items:
        cand = str(item).casefold().strip(" .")
        if target == cand or target in cand or cand in target:
            return True
    return False


def qc_source(db: IndexDB, source_id: int) -> dict[str, Any]:
    src = db.conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
    shots = list(db.conn.execute("SELECT * FROM shots WHERE source_id=? ORDER BY shot_index", (source_id,)))
    clips = list(db.conn.execute(
        "SELECT c.*, s.shot_uid FROM clips c JOIN shots s ON s.id=c.shot_id WHERE c.source_id=? ORDER BY c.clip_index",
        (source_id,),
    ))
    issues: list[str] = []
    warnings: list[str] = []
    analyses: dict[int, dict[str, Any]] = {}

    if not shots:
        issues.append("No shots recorded")
    else:
        if not _close(float(shots[0]["start_time"]), 0.0):
            issues.append("Physical shots do not start near 0")
        if not _close(float(shots[-1]["end_time"]), float(src["duration"])):
            issues.append("Physical shots do not end near source duration")
        for prev, cur in zip(shots, shots[1:]):
            if not _close(float(prev["end_time"]), float(cur["start_time"])):
                issues.append(f"Shot gap/overlap between {prev['shot_uid']} and {cur['shot_uid']}")

    for clip in clips:
        if float(clip["start_time"]) < -1e-6:
            issues.append(f"Negative clip start: {clip['clip_uid']}")
        if float(clip["end_time"]) > float(src["duration"]) + 0.08:
            issues.append(f"Clip exceeds source duration: {clip['clip_uid']}")
        if float(clip["end_time"]) <= float(clip["start_time"]):
            issues.append(f"Invalid clip duration: {clip['clip_uid']}")
        if clip["status"] != "INDEXED":
            continue
        try:
            analysis = json.loads(clip["analysis_json"] or "{}")
        except json.JSONDecodeError:
            issues.append(f"Invalid analysis JSON: {clip['clip_uid']}")
            continue
        analyses[int(clip["id"])] = analysis
        if not analysis.get("description"):
            issues.append(f"Missing description: {clip['clip_uid']}")
        if analysis.get("shot_type") not in SHOT_TYPES:
            issues.append(f"Non-normalized shot_type: {clip['clip_uid']}")
        if analysis.get("camera_motion") not in CAMERA_MOTIONS:
            issues.append(f"Non-normalized camera_motion: {clip['clip_uid']}")
        if analysis.get("content_type") not in CONTENT_TYPES:
            issues.append(f"Invalid content_type: {clip['clip_uid']}")
        if analysis.get("confidence_level") not in CONFIDENCE_LEVELS:
            issues.append(f"Invalid confidence_level: {clip['clip_uid']}")
        score = analysis.get("analysis_confidence")
        if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            issues.append(f"Invalid analysis_confidence: {clip['clip_uid']}")
        if not isinstance(analysis.get("confidence_basis"), list) or not analysis.get("confidence_basis"):
            issues.append(f"Missing deterministic confidence_basis: {clip['clip_uid']}")
        if analysis.get("exclusion_reason") not in EXCLUSION_REASONS:
            issues.append(f"Invalid exclusion_reason: {clip['clip_uid']}")
        if int(clip["analysis_version"] or 0) >= 4:
            if analysis.get("subject_focus") not in SUBJECT_FOCUS:
                issues.append(f"Invalid subject_focus: {clip['clip_uid']}")
            if analysis.get("shot_angle") not in SHOT_ANGLES:
                issues.append(f"Invalid shot_angle: {clip['clip_uid']}")
            if analysis.get("editorial_role") not in EDITORIAL_ROLES:
                issues.append(f"Invalid editorial_role: {clip['clip_uid']}")
            if analysis.get("visual_energy") not in VISUAL_ENERGIES:
                issues.append(f"Invalid visual_energy: {clip['clip_uid']}")
            if analysis.get("lighting_style") not in LIGHTING_STYLES:
                issues.append(f"Invalid lighting_style: {clip['clip_uid']}")
            if analysis.get("composition") not in COMPOSITIONS:
                issues.append(f"Invalid composition: {clip['clip_uid']}")
            family = analysis.get("visual_family")
            if not isinstance(family, str) or not family.strip():
                issues.append(f"Missing visual_family: {clip['clip_uid']}")
            for field_name in ("visual_quality_score", "editorial_usefulness_score"):
                score_v = analysis.get(field_name)
                if not isinstance(score_v, (int, float)) or not 0.0 <= float(score_v) <= 1.0:
                    issues.append(f"Invalid {field_name}: {clip['clip_uid']}")
        for flag in analysis.get("quality_flags", []):
            if flag not in QUALITY_FLAGS:
                issues.append(f"Invalid quality flag {flag!r}: {clip['clip_uid']}")
        for item in analysis.get("visible_text", []):
            low = str(item).casefold()
            if low.startswith("no visible text") or low.startswith("no readable text"):
                issues.append(f"Placeholder stored in visible_text: {clip['clip_uid']}")
        for item in analysis.get("observed_details", []):
            if _UNCERTAINTY_RE.search(str(item)):
                issues.append(f"Inference leaked into observed_details: {clip['clip_uid']}")
        for item in analysis.get("uncertain_inferences", []):
            if not _UNCERTAINTY_RE.search(str(item)):
                warnings.append(f"Inference lacks explicit uncertainty language: {clip['clip_uid']}")
                break

        exclusion = analysis.get("exclusion_reason", "none")
        usable = bool(analysis.get("usable", True))
        if exclusion != "none" and usable:
            issues.append(f"Excluded content marked usable: {clip['clip_uid']}")
        if analysis.get("content_type") in {"end_screen", "intro_card", "credits", "logo_screen", "text_screen"} and usable:
            issues.append(f"Non-editorial screen marked usable: {clip['clip_uid']}")

        primary_product = str(analysis.get("primary_product", "none"))
        objects = analysis.get("objects", []) if isinstance(analysis.get("objects"), list) else []
        if primary_product.casefold() not in {"none", "unknown"} and not _contains_entity(objects, primary_product):
            issues.append(f"Primary product missing from objects: {clip['clip_uid']}")

        primary_subject = str(analysis.get("primary_subject", "unknown"))
        people = analysis.get("people", []) if isinstance(analysis.get("people"), list) else []
        if _PERSON_RE.search(primary_subject) and not _contains_entity(people, primary_subject):
            issues.append(f"Visible person/fragment missing from people: {clip['clip_uid']}")

        if analysis.get("content_type") in {"product_closeup", "product_in_box"} and analysis.get("shot_type") in {"medium", "medium_wide"} and not people:
            issues.append(f"Product-only close visual has inconsistent shot_type: {clip['clip_uid']}")

        times = json.loads(clip["representative_times_json"] or "[]")
        if not times:
            issues.append(f"No representative frames: {clip['clip_uid']}")
        for t in times:
            if not (float(clip["start_time"]) - 0.01 <= float(t) <= float(clip["end_time"]) + 0.01):
                issues.append(f"Representative frame outside clip: {clip['clip_uid']} @ {t}")

    # Semantic-quality diagnostics: warnings do not fail a technically valid index.
    by_shot: dict[str, list[Any]] = defaultdict(list)
    for clip in clips:
        by_shot[str(clip["shot_uid"])].append(clip)
    for shot_uid, rows in by_shot.items():
        if len(rows) >= 6:
            avg = sum(float(r["duration"]) for r in rows) / len(rows)
            if avg < 4.0:
                warnings.append(f"Possible over-segmentation: {shot_uid} has {len(rows)} clips averaging {avg:.2f}s")
        for a, b in zip(rows, rows[1:]):
            aa = analyses.get(int(a["id"]))
            bb = analyses.get(int(b["id"]))
            if aa and bb and _jaccard(_tokens(aa), _tokens(bb)) >= 0.82:
                warnings.append(f"Adjacent semantic clips are highly similar: {a['clip_uid']} / {b['clip_uid']}")

    confidence_counts = Counter(
        a.get("confidence_level") for a in analyses.values() if a.get("confidence_level") in CONFIDENCE_LEVELS
    )
    confidence_scores = [float(a["analysis_confidence"]) for a in analyses.values() if isinstance(a.get("analysis_confidence"), (int, float))]
    total_conf = sum(confidence_counts.values())
    if total_conf >= 8 and confidence_scores:
        most_common, n = confidence_counts.most_common(1)[0]
        score_span = max(confidence_scores) - min(confidence_scores)
        # Warn only when BOTH labels and numeric scores are effectively uniform.
        if n / total_conf >= 0.90 and score_span < 0.06:
            warnings.append(
                f"Confidence is unusually uniform: {n}/{total_conf} clips are {most_common}; numeric span={score_span:.2f}"
            )

    indexed = sum(1 for c in clips if c["status"] == "INDEXED")
    unusable = sum(1 for a in analyses.values() if not bool(a.get("usable", True)))
    face_counts = Counter(
        (c["face_status"] or "not_scanned") for c in clips if c["status"] == "INDEXED"
    )
    confidence_summary = {
        "min": round(min(confidence_scores), 2) if confidence_scores else None,
        "max": round(max(confidence_scores), 2) if confidence_scores else None,
        "average": round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else None,
    }
    editorial_current = [a for c_id, a in analyses.items() if int(next((c["analysis_version"] or 0 for c in clips if int(c["id"]) == c_id), 0)) >= 4]
    editorial_roles = Counter(a.get("editorial_role", "other") for a in editorial_current)
    visual_families = {str(a.get("visual_family")) for a in editorial_current if a.get("visual_family")}

    return {
        "source_uid": src["source_uid"],
        "source_video": src["relative_path"],
        "duration": src["duration"],
        "shot_count": len(shots),
        "clip_count": len(clips),
        "indexed_clip_count": indexed,
        "unusable_clip_count": unusable,
        "failed_or_pending_clip_count": len(clips) - indexed,
        "confidence_distribution": dict(confidence_counts),
        "confidence_score_summary": confidence_summary,
        "face_status_distribution": dict(face_counts),
        "editorial_profile_clip_count": len(editorial_current),
        "editorial_role_distribution": dict(editorial_roles),
        "visual_family_count": len(visual_families),
        "passed": not issues and len(clips) > 0 and indexed == len(clips),
        "issues": issues,
        "warnings": warnings,
    }


def write_qc_report(db: IndexDB, output_path: Path) -> dict[str, Any]:
    reports = [qc_source(db, int(src["id"])) for src in db.all_sources()]
    report = {
        "passed": bool(reports) and all(r["passed"] for r in reports),
        "source_count": len(reports),
        "warning_count": sum(len(r.get("warnings", [])) for r in reports),
        "sources": reports,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
