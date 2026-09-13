from __future__ import annotations
import base64, math, re, subprocess
from pathlib import Path
from .ollama import chat

FACE_GATE_VERSION='face-v2'
SCHEMA={"type":"object","properties":{"status":{"type":"string","enum":["no_face","face_visible","uncertain"]},"reason":{"type":"string"}},"required":["status","reason"],"additionalProperties":False}
PERSON_RE=re.compile(r'\b(person|people|man|woman|presenter|host|human|face|head)\b',re.I)

def _dense_needed(clip):
    return bool(clip.people) or clip.content_type in {'presenter_demo','talking_head'} or bool(PERSON_RE.search(clip.primary_subject or ''))

def _extract_frame(source:Path,t:float,out:Path,width:int):
    """Extract one RGB PNG frame.

    PNG avoids FFmpeg MJPEG/yuv-range encoder failures seen on some source files.
    """
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([
        'ffmpeg','-y','-v','error','-ss',f'{t:.3f}','-i',str(source),
        '-frames:v','1','-vf',f'scale={width}:-2:flags=lanczos',
        '-c:v','png','-pix_fmt','rgb24','-threads','1',str(out)
    ],check=True)

def _interior_times(clip,n:int):
    dur=max(0.01,float(clip.duration))
    # Avoid exact shot/clip boundaries, which may be a cut frame or EOF.
    margin=min(0.12,max(0.02,dur*0.02))
    a=float(clip.start_time)+margin
    b=float(clip.end_time)-margin
    if b<=a:
        return [(float(clip.start_time)+float(clip.end_time))/2]
    if n<=1:return [(a+b)/2]
    return [a+(b-a)*(i/(n-1)) for i in range(n)]

def _evidence(cfg,clip,cache):
    existing=[Path(p) for p in clip.representative_paths if Path(p).exists()]
    if not _dense_needed(clip): return existing[:5]
    source=Path(clip.source_path)
    if not source.exists(): return existing[:5]
    interval=max(.25,float(cfg.face_scan_interval_seconds)); dur=max(0.01,clip.duration)
    n=min(int(cfg.face_scan_max_frames),max(3,int(math.ceil(dur/interval))))
    times=_interior_times(clip,n)
    folder=cache.root/'face_frames'/FACE_GATE_VERSION/clip.clip_id
    outs=[]
    for i,t in enumerate(times):
        p=folder/f'f_{i:03d}_{int(t*1000):010d}.png'
        if not p.exists():
            try:_extract_frame(source,t,p,int(cfg.face_frame_width))
            except Exception: continue
        if p.exists() and p.stat().st_size>64:outs.append(p)
    # Existing Agent-1 evidence is still useful if a source-specific extraction fails.
    return outs or existing[:5]

def screen_clip(cfg,clip,cache):
    hit=cache.get_face(clip.clip_id,clip.analysis_hash,FACE_GATE_VERSION)
    if hit:return hit
    paths=_evidence(cfg,clip,cache)
    if not paths:
        status,reason='uncertain','No visual evidence available for strict face verification.'
        cache.put_face(clip.clip_id,clip.analysis_hash,status,reason,FACE_GATE_VERSION); return status,reason
    prompt=("Inspect EVERY supplied frame for a HARD faceless-video gate. "
            "Return face_visible if ANY human facial area is actually visible: full face, side/profile face, partial facial features, reflection, photo/screen face, or an occluded face with identifiable facial area. "
            "Return no_face when the frames show only products, hands, wrists, arms, torso/clothing, gloves, or the back of a head with NO facial area visible. The mere presence of a person outside the crop is NOT uncertainty. "
            "Return uncertain ONLY when an actual head/facial region is visible or plausibly visible in-frame but blur, crop, reflection, or occlusion prevents deciding whether facial features are present. "
            "Be conservative about real facial pixels, not about off-screen people.")
    overall='no_face'; reasons=[]
    bs=max(1,int(cfg.face_batch_size))
    for i in range(0,len(paths),bs):
        batch=paths[i:i+bs]; imgs=[base64.b64encode(p.read_bytes()).decode('ascii') for p in batch]
        try:
            data=chat(cfg.ollama_url,cfg.face_model,[{'role':'user','content':prompt,'images':imgs}],SCHEMA,cfg.face_num_predict,0.0)
            status=data.get('status','uncertain'); reason=data.get('reason','')
        except Exception as e:
            status,reason='uncertain',f'Face gate failed: {e}'
        reasons.append(f'batch {i//bs+1}: {status} - {reason}')
        if status=='face_visible': overall='face_visible'; break
        if status!='no_face': overall='uncertain'
    reason='; '.join(reasons)[:1800]
    cache.put_face(clip.clip_id,clip.analysis_hash,overall,reason,FACE_GATE_VERSION)
    return overall,reason
