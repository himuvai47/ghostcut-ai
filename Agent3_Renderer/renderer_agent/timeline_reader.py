from __future__ import annotations
from pathlib import Path
from .utils import read_json

def load_timeline(path: Path) -> dict:
    if not path.exists(): raise FileNotFoundError(f"Timeline not found: {path}")
    data=read_json(path)
    if not isinstance(data,dict): raise ValueError("timeline root must be an object")
    audio=data.get("audio") or {}
    if not audio.get("source"): raise ValueError("timeline.audio.source is required")
    if float(audio.get("duration") or 0)<=0: raise ValueError("timeline.audio.duration must be > 0")
    segs=data.get("segments")
    if not isinstance(segs,list) or not segs: raise ValueError("timeline.segments must be a non-empty list")
    return data

def flatten_clips(timeline: dict) -> list[dict]:
    out=[]
    for seg in timeline["segments"]:
        if seg.get("status")!="matched": continue
        for c in seg.get("clips",[]):
            item=dict(c)
            item["segment_id"]=seg.get("segment_id")
            out.append(item)
    return sorted(out,key=lambda x:(float(x["timeline_start"]),float(x["timeline_end"])))

def validate_structure(timeline: dict, *, tolerance: float=0.001) -> list[str]:
    issues=[]
    clips=flatten_clips(timeline)
    prev_end=0.0
    for i,c in enumerate(clips):
        try:
            ss=float(c["source_start"]); se=float(c["source_end"]); ts=float(c["timeline_start"]); te=float(c["timeline_end"])
        except Exception:
            issues.append(f"clip {i+1}: invalid/missing numeric timestamps"); continue
        if ss<0: issues.append(f"clip {i+1}: source_start < 0")
        if se<=ss: issues.append(f"clip {i+1}: source_end <= source_start")
        if ts<0: issues.append(f"clip {i+1}: timeline_start < 0")
        if te<=ts: issues.append(f"clip {i+1}: timeline_end <= timeline_start")
        if ts < prev_end-tolerance: issues.append(f"clip {i+1}: overlaps previous timeline clip")
        prev_end=max(prev_end,te)
    return issues
