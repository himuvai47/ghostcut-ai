from __future__ import annotations
from pathlib import Path
import json
from .utils import file_fingerprint, sha256_text

def clip_cache_key(item:dict,cfg,encoder:str)->str:
    src=Path(item["source"])
    payload={"source":file_fingerprint(src),"source_start":round(float(item["source_start"]),6),"source_end":round(float(item["source_end"]),6),"frames":int(item["frame_count"]),"width":cfg.width,"height":cfg.height,"fps":cfg.fps,"encoder":encoder,"pixel_format":cfg.pixel_format,"nvenc_preset":cfg.nvenc_preset,"nvenc_cq":cfg.nvenc_cq,"x264_preset":cfg.x264_preset,"x264_crf":cfg.x264_crf}
    return sha256_text(json.dumps(payload,sort_keys=True,separators=(",",":")))
