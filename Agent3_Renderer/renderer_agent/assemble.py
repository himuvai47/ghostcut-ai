from __future__ import annotations
from pathlib import Path
from .utils import ffconcat_quote, run

def write_concat_manifest(paths:list[Path], manifest:Path):
    manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text("ffconcat version 1.0\n"+"".join(f"file {ffconcat_quote(p)}\n" for p in paths),encoding="utf-8")

def concatenate(paths:list[Path],manifest:Path,out:Path):
    if not paths: raise RuntimeError("No rendered video items to concatenate")
    write_concat_manifest(paths,manifest)
    tmp=out.with_suffix('.tmp.mp4')
    if tmp.exists(): tmp.unlink()
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-f","concat","-safe","0","-i",str(manifest),"-c","copy","-movflags","+faststart",str(tmp)])
    tmp.replace(out)

def mux_voiceover(video:Path,audio:Path,target_duration:float,cfg,out:Path):
    tmp=out.with_suffix('.tmp.mp4')
    if tmp.exists(): tmp.unlink()
    run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(video),"-i",str(audio),"-map","0:v:0","-map","1:a:0","-c:v","copy","-c:a",cfg.audio_codec,"-b:a",cfg.audio_bitrate,"-ar",str(cfg.audio_sample_rate),"-t",f"{target_duration:.6f}","-movflags","+faststart",str(tmp)])
    tmp.replace(out)
