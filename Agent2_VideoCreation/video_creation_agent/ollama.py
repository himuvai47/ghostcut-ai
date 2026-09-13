from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any


def _post(url: str, payload: dict, timeout: int = 600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _validate_schema(value: Any, schema: dict, path: str = "$") -> None:
    """Small local validator for the JSON-schema subset Agent 2 uses.

    This intentionally avoids adding a dependency just for runtime response checks.
    Supported: object/array/string/boolean/number/integer/null, required, enum,
    items, properties and additionalProperties=false.
    """
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value {value!r} is not in enum {schema['enum']!r}")

    typ = schema.get("type")
    if typ == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected object")
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}: missing required field {key!r}")
        if schema.get("additionalProperties") is False:
            extras = set(value) - set(props)
            if extras:
                raise ValueError(f"{path}: unexpected field(s): {sorted(extras)!r}")
        for key, child in props.items():
            if key in value:
                _validate_schema(value[key], child, f"{path}.{key}")
    elif typ == "array":
        if not isinstance(value, list):
            raise ValueError(f"{path}: expected array")
        child = schema.get("items")
        if child:
            for i, item in enumerate(value):
                _validate_schema(item, child, f"{path}[{i}]")
    elif typ == "string":
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
    elif typ == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{path}: expected boolean")
    elif typ == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{path}: expected integer")
    elif typ == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{path}: expected number")
    elif typ == "null":
        if value is not None:
            raise ValueError(f"{path}: expected null")


def _decode_json(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty response content")
    s = text.strip()

    # Some models wrap otherwise-valid JSON in Markdown despite explicit instructions.
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.I | re.S)
    if m:
        s = m.group(1).strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError as first:
        # Final compatibility fallback: accept one JSON object embedded in harmless
        # leading/trailing prose, but never attempt to repair malformed JSON.
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(s[a : b + 1])
            except json.JSONDecodeError:
                pass
        raise first


def _schema_prompt(schema: dict) -> str:
    compact = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return (
        "\n\nOUTPUT CONTRACT (mandatory): Return ONLY one valid JSON object. "
        "Do not use Markdown fences, commentary, or prose outside the JSON. "
        "The JSON must match this schema exactly:\n" + compact
    )


def _with_schema_instruction(messages: list[dict], schema: dict) -> list[dict]:
    out = [dict(m) for m in messages]
    instruction = _schema_prompt(schema)
    # Append to the final user message so image arrays and role ordering are preserved.
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") == "user":
            out[i] = dict(out[i])
            out[i]["content"] = str(out[i].get("content", "")) + instruction
            return out
    out.append({"role": "user", "content": instruction.lstrip()})
    return out


def _parse_and_validate(data: dict, schema: dict) -> Any:
    msg = data.get("message") or {}
    content = msg.get("content") or ""
    if content.strip():
        value = _decode_json(content)
        _validate_schema(value, schema)
        return value

    # A few Ollama/Qwen regressions have routed output to `thinking` while leaving
    # content empty. Only accept it if the ENTIRE thinking field is itself valid
    # schema-conforming JSON; ordinary chain-of-thought is never treated as data.
    thinking = msg.get("thinking") or ""
    if thinking.strip():
        try:
            value = json.loads(thinking.strip())
            _validate_schema(value, schema)
            return value
        except Exception:
            pass

    raise ValueError(
        "empty response content "
        f"(thinking_chars={len(thinking)}, done_reason={data.get('done_reason')!r}, "
        f"eval_count={data.get('eval_count')!r})"
    )


def chat(
    base: str,
    model: str,
    messages: list[dict],
    schema: dict,
    num_predict: int = 4096,
    temperature: float = 0.0,
):
    """Robust Ollama structured chat for Qwen3.5 and vision models.

    Qwen3.5/Ollama versions have differed in how thinking interacts with structured
    output. Instead of repeating the same failed request three times, use three
    distinct compatibility modes:
      1) schema + low thinking (best editorial reasoning when supported)
      2) schema + thinking disabled (prevents a thinking trace consuming the turn)
      3) thinking disabled + prompt-enforced JSON (survives format regressions)
    Every successful result is parsed and validated locally against `schema`.
    """
    url = base.rstrip("/") + "/api/chat"
    last: Exception | None = None
    diagnostics: list[str] = []

    modes = [
        {
            "name": "schema_low_think",
            "think": "low",
            "format": schema,
            "messages": messages,
            "extra_tokens": 1024,
        },
        {
            "name": "schema_no_think",
            "think": False,
            "format": schema,
            "messages": _with_schema_instruction(messages, schema),
            "extra_tokens": 0,
        },
        {
            "name": "prompt_json_no_think",
            "think": False,
            "format": None,
            "messages": _with_schema_instruction(messages, schema),
            "extra_tokens": 1024,
        },
    ]

    for mode in modes:
        try:
            payload = {
                "model": model,
                "messages": mode["messages"],
                "stream": False,
                "think": mode["think"],
                "options": {
                    "temperature": temperature,
                    "num_predict": int(num_predict) + int(mode["extra_tokens"]),
                },
            }
            if mode["format"] is not None:
                payload["format"] = mode["format"]

            data = _post(url, payload)
            return _parse_and_validate(data, schema)
        except Exception as e:
            last = e
            diagnostics.append(f"{mode['name']}: {type(e).__name__}: {e}")

    detail = " | ".join(diagnostics)
    raise RuntimeError(f"Ollama structured response failed in all compatibility modes: {detail}") from last



def chat_fast(
    base: str,
    model: str,
    messages: list[dict],
    schema: dict,
    num_predict: int = 1024,
    temperature: float = 0.0,
):
    """Fast structured chat for simple classification/ranking tasks.

    Unlike :func:`chat`, this deliberately skips model thinking and tries only
    compact JSON modes. Agent 2 uses it for repetitive candidate ranking and
    lightweight editorial checks where chain-of-thought adds latency but little
    value. Planner/script understanding continues to use the full compatibility
    path with low thinking.
    """
    url = base.rstrip("/") + "/api/chat"
    last: Exception | None = None
    diagnostics: list[str] = []
    modes = [
        {
            "name": "schema_no_think",
            "format": schema,
            "messages": _with_schema_instruction(messages, schema),
        },
        {
            "name": "prompt_json_no_think",
            "format": None,
            "messages": _with_schema_instruction(messages, schema),
        },
    ]
    for mode in modes:
        try:
            payload = {
                "model": model,
                "messages": mode["messages"],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": int(num_predict),
                },
            }
            if mode["format"] is not None:
                payload["format"] = mode["format"]
            data = _post(url, payload)
            return _parse_and_validate(data, schema)
        except Exception as e:
            last = e
            diagnostics.append(f"{mode['name']}: {type(e).__name__}: {e}")
    detail = " | ".join(diagnostics)
    raise RuntimeError(f"Ollama fast structured response failed in all compatibility modes: {detail}") from last

def embed(base: str, model: str, texts: list[str]):
    data = _post(base.rstrip("/") + "/api/embed", {"model": model, "input": texts, "truncate": True})
    return data["embeddings"]
