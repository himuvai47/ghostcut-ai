from __future__ import annotations

from hashlib import sha1
from pathlib import Path
import re


def stable_source_uid(relative_path: str) -> str:
    digest = sha1(relative_path.replace("\\", "/").lower().encode("utf-8")).hexdigest()[:12]
    return f"src_{digest}"


def safe_name(value: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (cleaned or "item")[:max_len]


def relpath_for_display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
