from __future__ import annotations
import json, sqlite3
from pathlib import Path
from .settings import GhostCutSettings
from .utils import read_json


def face_metadata_status(workspace: Path) -> dict:
    db=workspace/"footage_index.db"
    if not db.exists(): return {"ready":False,"reason":"footage_index.db missing","counts":{}}
    try:
        con=sqlite3.connect(str(db)); con.row_factory=sqlite3.Row
        cols={r[1] for r in con.execute("PRAGMA table_info(clips)").fetchall()}
        if "face_status" not in cols or "face_scan_version" not in cols:
            return {"ready":False,"reason":"face metadata missing","counts":{}}
        rows=con.execute("SELECT face_status, COUNT(*) n, MIN(COALESCE(face_scan_version,0)) minv FROM clips GROUP BY face_status").fetchall()
        counts={str(r["face_status"]):int(r["n"]) for r in rows}
        minv=min([int(r["minv"] or 0) for r in rows],default=0)
        ready=all(k in {"no_face","face_visible","uncertain"} for k in counts) and minv>=2
        return {"ready":ready,"reason":"ready" if ready else f"face scan version {minv}; v2+ required","counts":counts,"min_version":minv}
    except Exception as e:
        return {"ready":False,"reason":str(e),"counts":{}}
    finally:
        try: con.close()
        except Exception: pass


def library_info(workspace: Path) -> dict:
    qc=read_json(workspace/"qc_report.json",{}) or {}
    info={"name":workspace.name,"path":str(workspace.resolve()),"qc_passed":bool(qc.get("passed",False)),"clip_count":0,"source_count":0,"face":face_metadata_status(workspace)}
    db=workspace/"footage_index.db"
    if db.exists():
        try:
            con=sqlite3.connect(str(db))
            info["clip_count"]=int(con.execute("SELECT COUNT(*) FROM clips").fetchone()[0])
            # Support old/new source table names defensively.
            tables={r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if "sources" in tables: info["source_count"]=int(con.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
            con.close()
        except Exception: pass
    return info


def discover_libraries(settings: GhostCutSettings) -> list[dict]:
    root=Path(settings.indexes_root); root.mkdir(parents=True,exist_ok=True)
    out=[]
    if (root/"footage_index.db").exists(): out.append(library_info(root))
    for p in root.iterdir():
        if p.is_dir() and (p/"footage_index.db").exists(): out.append(library_info(p))
    return sorted(out,key=lambda x:x["name"].lower())
