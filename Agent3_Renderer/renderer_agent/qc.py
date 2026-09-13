from __future__ import annotations
from pathlib import Path
from .probe import probe,duration_seconds,video_stream,audio_stream,parse_rate

def validate_output(path:Path, plan:dict,cfg)->dict:
    issues=[]
    if not path.exists(): return {"passed":False,"issues":["output file missing"]}
    info=probe(path); vs=video_stream(info); aud=audio_stream(info); dur=duration_seconds(info)
    if not vs: issues.append("video stream missing")
    if not aud: issues.append("audio stream missing")
    if vs:
        if int(vs.get("width") or 0)!=cfg.width or int(vs.get("height") or 0)!=cfg.height:
            issues.append(f"resolution mismatch: {vs.get('width')}x{vs.get('height')}")
        fps=parse_rate(vs.get("avg_frame_rate") or vs.get("r_frame_rate"))
        if abs(fps-cfg.fps)>0.05: issues.append(f"fps mismatch: {fps:.4f}")
    else: fps=0.0
    target=float(plan["target_duration"]); drift=abs(dur-target)
    if drift>cfg.duration_tolerance_seconds:
        issues.append(f"duration drift {drift:.3f}s exceeds tolerance {cfg.duration_tolerance_seconds:.3f}s")
    # Render plan must cover all target frames after explicit gap fillers.
    frame_sum=sum(int(i["frame_count"]) for i in plan["items"])
    if frame_sum!=int(plan["target_frames"]): issues.append(f"render plan frame coverage mismatch: {frame_sum} != {plan['target_frames']}")
    return {"passed":not issues,"issues":issues,"duration":round(dur,3),"expected_duration":round(target,4),"duration_drift":round(drift,4),"resolution":f"{vs.get('width')}x{vs.get('height')}" if vs else None,"fps":round(fps,3),"video_codec":vs.get("codec_name") if vs else None,"audio_codec":aud.get("codec_name") if aud else None,"frame_coverage":frame_sum}
