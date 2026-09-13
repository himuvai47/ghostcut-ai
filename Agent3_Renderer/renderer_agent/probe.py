from __future__ import annotations
import json
from pathlib import Path
from .utils import run

def probe(path: Path) -> dict:
    p=run(["ffprobe","-v","error","-show_streams","-show_format","-of","json",str(path)])
    return json.loads(p.stdout)

def duration_seconds(info: dict) -> float:
    d=(info.get("format") or {}).get("duration")
    if d is not None:
        return float(d)
    vals=[]
    for s in info.get("streams",[]):
        if s.get("duration") is not None:
            vals.append(float(s["duration"]))
    return max(vals) if vals else 0.0

def video_stream(info: dict):
    return next((s for s in info.get("streams",[]) if s.get("codec_type")=="video"),None)

def audio_stream(info: dict):
    return next((s for s in info.get("streams",[]) if s.get("codec_type")=="audio"),None)

def parse_rate(text: str|None) -> float:
    if not text or text in {"0/0","N/A"}: return 0.0
    if "/" in text:
        a,b=text.split("/",1)
        return float(a)/float(b) if float(b) else 0.0
    return float(text)
