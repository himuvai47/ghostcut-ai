from __future__ import annotations

from hashlib import sha256
from pathlib import Path


def fingerprint_file(path: Path, mode: str = "sampled", sample_bytes: int = 4 * 1024 * 1024) -> str:
    stat = path.stat()
    h = sha256()
    h.update(str(stat.st_size).encode("ascii"))
    if mode == "full":
        with path.open("rb") as f:
            while True:
                chunk = f.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return f"full:{h.hexdigest()}"

    if mode != "sampled":
        raise ValueError(f"Unsupported fingerprint mode: {mode}")

    size = stat.st_size
    offsets = [0]
    if size > sample_bytes:
        offsets.append(max(0, size // 2 - sample_bytes // 2))
        offsets.append(max(0, size - sample_bytes))
    with path.open("rb") as f:
        for offset in sorted(set(offsets)):
            f.seek(offset)
            h.update(offset.to_bytes(8, "little", signed=False))
            h.update(f.read(sample_bytes))
    return f"sampled:{h.hexdigest()}"
