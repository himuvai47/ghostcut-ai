from __future__ import annotations
from pathlib import Path
from .cache import clip_cache_key
from .encoder import video_encode_args
from .utils import run

def _vf(cfg, source_duration:float, frame_count:int)->str:
    # trim protects Agent 2's exact source window; tpad only compensates for decoder/frame rounding.
    return (f"trim=duration={source_duration:.6f},setpts=PTS-STARTPTS,"
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,"
            f"crop={cfg.width}:{cfg.height},fps={cfg.fps},"
            f"tpad=stop_mode=clone:stop_duration=1,trim=end_frame={frame_count},"
            f"setpts=PTS-STARTPTS,format={cfg.pixel_format}")

def render_clip(item:dict,cfg,encoder:str,cache_dir:Path,force:bool=False)->tuple[Path,bool]:
    cache_dir.mkdir(parents=True,exist_ok=True)
    key=clip_cache_key(item,cfg,encoder)
    out=cache_dir/f"{key}.mp4"
    if cfg.cache_enabled and out.exists() and not force:
        return out,True
    tmp=out.with_suffix('.tmp.mp4')
    if tmp.exists(): tmp.unlink()
    source_duration=max(0.001,float(item["source_end"])-float(item["source_start"]))
    cmd=["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",f"{float(item['source_start']):.6f}","-i",item["source"],"-an","-vf",_vf(cfg,source_duration,int(item["frame_count"])),"-frames:v",str(int(item["frame_count"])),"-r",str(cfg.fps),*video_encode_args(cfg,encoder),"-movflags","+faststart",str(tmp)]
    run(cmd)
    tmp.replace(out)
    return out,False

def render_freeze_gap(source_video:Path,gap_item:dict,cfg,encoder:str,out:Path, *, use_first_frame=False):
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp_img=out.with_suffix('.freeze.png')
    if tmp_img.exists(): tmp_img.unlink()
    if use_first_frame:
        cmd1=["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(source_video),"-frames:v","1",str(tmp_img)]
    else:
        cmd1=["ffmpeg","-hide_banner","-loglevel","error","-y","-sseof","-0.2","-i",str(source_video),"-frames:v","1",str(tmp_img)]
    run(cmd1)
    frames=int(gap_item["frame_count"]); dur=frames/cfg.fps
    cmd2=["ffmpeg","-hide_banner","-loglevel","error","-y","-loop","1","-i",str(tmp_img),"-an","-vf",f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,crop={cfg.width}:{cfg.height},fps={cfg.fps},format={cfg.pixel_format}","-frames:v",str(frames),"-t",f"{dur:.6f}","-r",str(cfg.fps),*video_encode_args(cfg,encoder),str(out)]
    try: run(cmd2)
    finally:
        if tmp_img.exists(): tmp_img.unlink()
