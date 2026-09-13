from __future__ import annotations
import json, shutil, urllib.request
from pathlib import Path
from .settings import GhostCutSettings


def _exists(root: str, name: str) -> bool:
    return (Path(root)/name).exists()


def _venv(root: str) -> bool:
    p=Path(root)/".venv"/"Scripts"/"python.exe"
    return p.exists()


def ollama_status(url: str) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/")+"/api/tags",timeout=2.5) as r:
            data=json.loads(r.read().decode("utf-8"))
        models=[str(x.get("name","")) for x in data.get("models",[])]
        return {"ok":True,"models":models}
    except Exception as e:
        return {"ok":False,"models":[],"error":str(e)}


def health(settings: GhostCutSettings) -> dict:
    ol=ollama_status(settings.ollama_url)
    a1=_exists(settings.agent1_root,"RUN_AGENT1.ps1") and _venv(settings.agent1_root)
    a2=_exists(settings.agent2_root,"RUN_AGENT2.ps1") and _venv(settings.agent2_root)
    a3=_exists(settings.agent3_root,"RUN_AGENT3.ps1") and _venv(settings.agent3_root)
    ff=bool(shutil.which("ffmpeg")); fp=bool(shutil.which("ffprobe"))
    models=ol.get("models",[])
    def has(prefix): return any(m==prefix or m.startswith(prefix+":") for m in models)
    return {
        "passed": bool(a1 and a2 and a3 and ff and fp and ol["ok"]),
        "agents":{"agent1":a1,"agent2":a2,"agent3":a3},
        "ffmpeg":ff,"ffprobe":fp,"ollama":ol["ok"],
        "models":{"qwen2.5vl":has("qwen2.5vl"),"qwen3.5":has("qwen3.5"),"nomic-embed-text":has("nomic-embed-text")},
        "paths":{"agent1":settings.agent1_root,"agent2":settings.agent2_root,"agent3":settings.agent3_root,"projects":settings.projects_root,"indexes":settings.indexes_root},
    }
