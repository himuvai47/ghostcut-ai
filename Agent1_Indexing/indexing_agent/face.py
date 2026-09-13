from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any
from urllib import error, request

from .config import IndexerConfig


FACE_SCAN_VERSION = 2
FACE_STATUSES = ("no_face", "face_visible", "uncertain")

FACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "face_status": {"type": "string", "enum": list(FACE_STATUSES)},
        "evidence_frame_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
            "maxItems": 5,
        },
        "reason": {"type": "string", "maxLength": 220},
    },
    "required": ["face_status", "evidence_frame_indices", "reason"],
    "additionalProperties": False,
}

FACE_SYSTEM_PROMPT = """You are a face-presence checker for source footage.
Inspect the supplied representative frames from ONE continuous semantic clip.
Answer only whether a HUMAN FACE is visibly present in any supplied frame.

Rules:
- Hands, arms, torso, clothing, hair, the back of a head, and a cropped body WITHOUT visible facial features do NOT count as a face.
- A clear frontal face, side-profile face, or clearly visible facial region DOES count as face_visible.
- A tiny, heavily blurred, reflected, obscured, or partial region that might be a face but cannot be established should be uncertain.
- If no facial features are visible in any supplied frame, return no_face, even if hands or a person's body are visible.
- evidence_frame_indices are 1-based indices of frames that support face_visible or uncertain. For no_face return [].
- Keep the reason extremely short.
- Output only one JSON object with exactly these keys: face_status, evidence_frame_indices, reason.
"""


@dataclass(slots=True)
class FaceScanResult:
    face_status: str
    evidence_frame_indices: list[int]
    reason: str
    mode: str = "representative_frames_qwen25vl"

    def evidence_payload(self, times: list[float]) -> dict[str, Any]:
        evidence = []
        for idx in self.evidence_frame_indices:
            item: dict[str, Any] = {"frame_index": idx}
            if 1 <= idx <= len(times):
                item["timestamp"] = float(times[idx - 1])
            evidence.append(item)
        return {"reason": self.reason, "frames": evidence}


class FaceScanError(RuntimeError):
    pass


def _json_request(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        suffix = f": {detail[:800]}" if detail else ""
        raise FaceScanError(f"Ollama HTTP {exc.code}{suffix}") from exc
    except (error.URLError, TimeoutError) as exc:
        raise FaceScanError(f"Ollama request failed: {exc}") from exc
    try:
        value = json.loads(body)
    except json.JSONDecodeError as exc:
        raise FaceScanError(f"Ollama returned invalid JSON envelope: {body[:300]}") from exc
    if not isinstance(value, dict):
        raise FaceScanError("Ollama response envelope was not an object")
    return value


def _encode(paths: list[Path]) -> list[str]:
    return [base64.b64encode(p.read_bytes()).decode("ascii") for p in paths]


def _validate(data: Any, frame_count: int) -> FaceScanResult:
    if not isinstance(data, dict):
        raise ValueError("face response root must be an object")
    status = data.get("face_status")
    if status not in FACE_STATUSES:
        raise ValueError(f"invalid face_status: {status!r}")
    indices = data.get("evidence_frame_indices")
    if not isinstance(indices, list) or any(not isinstance(i, int) for i in indices):
        raise ValueError("evidence_frame_indices must be an integer array")
    indices = sorted({i for i in indices if 1 <= i <= frame_count})[:5]
    reason = data.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No additional reason supplied."
    reason = reason.strip()[:220]
    if status == "no_face":
        indices = []
    return FaceScanResult(status, indices, reason)


def scan_face_status(paths: list[Path], cfg: IndexerConfig) -> FaceScanResult:
    if not paths:
        raise FaceScanError("No representative frames available for face scan")
    missing = [str(p) for p in paths if not p.exists() or p.stat().st_size <= 0]
    if missing:
        raise FaceScanError(f"Representative frame missing: {missing[0]}")

    images = _encode(paths)
    payload = {
        "model": cfg.vision_model,
        "messages": [
            {"role": "system", "content": FACE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Check these {len(paths)} ordered representative frames for visible human faces.",
                "images": images,
            },
        ],
        "format": "json",
        "stream": False,
        "keep_alive": cfg.ollama_keep_alive,
        "options": {
            "temperature": 0,
            "num_ctx": cfg.ollama_num_ctx,
            "num_predict": 160,
        },
    }

    last: Exception | None = None
    attempts = min(2, max(1, cfg.model_retries))
    for attempt in range(1, attempts + 1):
        try:
            env = _json_request(cfg.ollama_url.rstrip("/") + "/api/chat", payload, cfg.ollama_timeout_sec)
            content = env.get("message", {}).get("content", "")
            if not isinstance(content, str) or not content.strip():
                raise FaceScanError("Ollama returned empty face-scan content")
            return _validate(json.loads(content), len(paths))
        except (FaceScanError, ValueError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < attempts:
                time.sleep(0.25)
    raise FaceScanError(f"Face scan failed after {attempts} attempts: {last}")
