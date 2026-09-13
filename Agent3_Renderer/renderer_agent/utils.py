from __future__ import annotations
from pathlib import Path
import hashlib, json, subprocess, os

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_json(path: Path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def file_fingerprint(path: Path) -> dict:
    st=path.stat()
    return {"path":str(path.resolve()),"size":st.st_size,"mtime_ns":st.st_mtime_ns}

def run(cmd:list[str], *, capture=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd,check=True,text=True,capture_output=capture,encoding="utf-8",errors="replace")

def which(exe:str):
    import shutil
    return shutil.which(exe)

def ffconcat_quote(path: Path) -> str:
    s=str(path.resolve()).replace("\\","/")
    return "'" + s.replace("'", "'\\''") + "'"
