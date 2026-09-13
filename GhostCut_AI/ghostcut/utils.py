from __future__ import annotations
import json, os, re, shutil, subprocess, sys, threading, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return value or "ghostcut-project"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, *, replace_retries: int = 12) -> None:
    """Write JSON atomically and tolerate short Windows sharing violations.

    GhostCut updates project state frequently while the UI is polling it.  A fixed
    ``project.json.tmp`` name lets concurrent writers collide, and Windows may also
    deny ``ReplaceFile``/rename for a few milliseconds while antivirus/indexers have
    the destination open.  Give every write its own temp file and retry only the
    atomic replace step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False)
    tmp = path.parent / (
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    tmp.write_text(payload, encoding="utf-8")
    last_error: PermissionError | None = None
    try:
        for attempt in range(max(1, int(replace_retries))):
            try:
                os.replace(tmp, path)
                return
            except PermissionError as exc:
                last_error = exc
                # Keep the total retry window short (~2.3 s with defaults), while
                # allowing common Windows Defender/Search indexer locks to clear.
                time.sleep(min(0.02 * (attempt + 1), 0.25))
        assert last_error is not None
        raise last_error
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def safe_copy(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return dst
    shutil.copy2(src, dst)
    return dst


def open_path(path: Path) -> None:
    path = path.resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
