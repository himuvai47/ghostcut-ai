from __future__ import annotations
from pathlib import Path
from .timeline_reader import validate_structure, flatten_clips
from .probe import probe, duration_seconds, video_stream, audio_stream
from .utils import which


def run_preflight(timeline: dict, workspace: Path) -> dict:
    issues = validate_structure(timeline)
    if not which("ffmpeg"):
        issues.append("ffmpeg not found in PATH")
    if not which("ffprobe"):
        issues.append("ffprobe not found in PATH")

    workspace.mkdir(parents=True, exist_ok=True)

    audio = Path(timeline["audio"]["source"])
    if not audio.exists():
        issues.append(f"voiceover missing: {audio}")
    else:
        try:
            ai = probe(audio)
            if not audio_stream(ai):
                issues.append(f"voiceover has no audio stream: {audio}")
        except Exception as e:
            issues.append(f"voiceover probe failed: {e}")

    seen = {}
    for c in flatten_clips(timeline):
        p = Path(c["source"])
        if p in seen:
            info = seen[p]
        else:
            if not p.exists():
                issues.append(f"source missing: {p}")
                continue
            try:
                info = probe(p)
                seen[p] = info
            except Exception as e:
                issues.append(f"source probe failed {p}: {e}")
                continue

        if not video_stream(info):
            issues.append(f"source has no video stream: {p}")
            continue

        dur = duration_seconds(info)
        if float(c["source_end"]) > dur + 0.15:
            issues.append(
                f"source range exceeds media duration: {p} "
                f"end={c['source_end']} duration={dur:.3f}"
            )

    if issues:
        raise RuntimeError("Preflight failed:\n - " + "\n - ".join(issues))

    return {
        "passed": True,
        "clips": len(flatten_clips(timeline)),
        "sources": len(seen),
        "audio": str(audio),
    }
