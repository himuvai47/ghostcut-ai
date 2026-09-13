from __future__ import annotations
import hashlib, json, logging, re
from pathlib import Path
from typing import Any

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def normalize_word(s: str) -> str:
    m = WORD_RE.findall(s.casefold())
    return "".join(m)

def script_words(text: str) -> list[str]:
    return WORD_RE.findall(text)

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def setup_logger(workspace: Path, verbose: bool = False) -> tuple[logging.Logger, Path]:
    from datetime import datetime
    workspace.mkdir(parents=True, exist_ok=True)
    logs = workspace / "logs"; logs.mkdir(parents=True, exist_ok=True)
    p = logs / f"agent2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(f"agent2.{id(workspace)}")
    logger.handlers.clear(); logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(levelname)s | %(message)s")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.FileHandler(p, encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger, p
