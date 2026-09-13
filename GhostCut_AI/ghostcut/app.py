from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import __version__
from .dialogs import choose_file, choose_folder
from .health import health
from .libraries import discover_libraries, library_info
from .projects import ProjectStore
from .runner import JobRegistry, PipelineRunner, RUN_MODES
from .settings import SettingsStore
from .utils import open_path

APP_ROOT=Path(__file__).resolve().parents[1]
settings_store=SettingsStore(APP_ROOT)
project_store=ProjectStore(settings_store.value)
registry=JobRegistry()
runner=PipelineRunner(settings_store.value,project_store,registry)
app=FastAPI(title="GhostCut AI",version=__version__,docs_url=None,redoc_url=None)

class ProjectIn(BaseModel):
    name:str
    script_path:str
    audio_path:str
    library_mode:str
    library_path:str=""
    footage_path:str=""
class SettingsIn(BaseModel):
    agent1_root:str|None=None; agent2_root:str|None=None; agent3_root:str|None=None; projects_root:str|None=None; indexes_root:str|None=None; ollama_url:str|None=None

@app.get("/api/meta")
def meta(): return {"name":"GhostCut AI","version":__version__}
@app.get("/api/health")
def api_health(): return health(settings_store.value)
@app.get("/api/settings")
def get_settings(): return asdict(settings_store.value)
@app.post("/api/settings")
def save_settings(payload:SettingsIn):
    global project_store,runner
    data={k:v for k,v in payload.model_dump().items() if v is not None}
    s=settings_store.update(data); project_store=ProjectStore(s); runner=PipelineRunner(s,project_store,registry); return asdict(s)
@app.get("/api/libraries")
def libraries(): return discover_libraries(settings_store.value)
@app.get("/api/library-info")
def libinfo(path:str):
    p=Path(path)
    if not p.exists(): raise HTTPException(404,"Library not found")
    return library_info(p)
@app.get("/api/projects")
def projects(): return [runner.reconcile_project(p).to_dict() for p in project_store.list()]
@app.post("/api/projects")
def create_project(payload:ProjectIn):
    try:return project_store.create(**payload.model_dump()).to_dict()
    except Exception as e: raise HTTPException(400,str(e))
@app.get("/api/projects/{project_id}")
def project(project_id:str):
    try:return runner.reconcile_project(project_store.get(project_id)).to_dict()
    except KeyError: raise HTTPException(404,"Project not found")

@app.post("/api/projects/{project_id}/run")
def run_project(project_id:str):
    return run_project_mode(project_id,"full")

@app.post("/api/projects/{project_id}/run/{mode}")
def run_project_mode(project_id:str,mode:str):
    if mode not in RUN_MODES: raise HTTPException(400,f"Unknown run mode: {mode}")
    try:
        p=runner.reconcile_project(project_store.get(project_id))
        return runner.start(p,mode).to_dict()
    except KeyError: raise HTTPException(404,"Project not found")
    except Exception as e: raise HTTPException(409,str(e))

@app.get("/api/projects/{project_id}/pipeline")
def project_pipeline(project_id:str):
    try:
        p=runner.reconcile_project(project_store.get(project_id))
        return {"project":p.to_dict(),"job":runner.project_snapshot(p),"artifacts":artifact_status(p)}
    except KeyError: raise HTTPException(404,"Project not found")

@app.get("/api/jobs/{job_id}")
def get_job(job_id:str):
    try:return registry.get(job_id).to_dict()
    except KeyError: raise HTTPException(404,"Job not found")
@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id:str):
    try:j=registry.get(job_id); runner.cancel(j); return j.to_dict()
    except KeyError: raise HTTPException(404,"Job not found")


def artifact_status(p):
    paths={
        "library":Path(p.library_path),
        "timeline":Path(p.agent2_workspace)/"timeline.json",
        "agent2_qc":Path(p.agent2_workspace)/"qc_report.json",
        "video":Path(p.rough_cut_path) if p.rough_cut_path else Path(p.agent3_workspace)/"rough_cut.mp4",
        "agent3_qc":Path(p.agent3_workspace)/"qc_report.json",
        "project":Path(p.root),
    }
    return {k:{"exists":v.exists(),"path":str(v)} for k,v in paths.items()}

@app.get("/api/projects/{project_id}/artifacts")
def artifacts(project_id:str):
    try:return artifact_status(project_store.get(project_id))
    except KeyError: raise HTTPException(404,"Project not found")

@app.get("/api/projects/{project_id}/video")
def project_video(project_id:str):
    try:p=project_store.get(project_id)
    except KeyError: raise HTTPException(404,"Project not found")
    path=Path(p.rough_cut_path) if p.rough_cut_path else Path(p.agent3_workspace)/"rough_cut.mp4"
    if not path.is_file(): raise HTTPException(404,"Rough cut not available")
    return FileResponse(path,media_type="video/mp4",filename="rough_cut.mp4")

@app.post("/api/projects/{project_id}/open-folder")
def open_folder(project_id:str):
    try:p=project_store.get(project_id); open_path(Path(p.root)); return {"ok":True}
    except KeyError: raise HTTPException(404,"Project not found")

@app.post("/api/projects/{project_id}/open-video")
def open_video(project_id:str):
    return open_artifact(project_id,"video")

@app.post("/api/projects/{project_id}/open-artifact/{kind}")
def open_artifact(project_id:str,kind:str):
    try:p=project_store.get(project_id)
    except KeyError: raise HTTPException(404,"Project not found")
    status=artifact_status(p)
    if kind not in status: raise HTTPException(400,"Unknown artifact")
    path=Path(status[kind]["path"])
    if not path.exists(): raise HTTPException(404,f"{kind} is not available yet")
    open_path(path); return {"ok":True}

@app.get("/api/dialog/file")
def dialog_file(kind:str="all",title:str="Choose file"):
    try:return {"path":choose_file(title,kind)}
    except Exception as e: raise HTTPException(500,f"Native file dialog failed: {e}")
@app.get("/api/dialog/folder")
def dialog_folder(title:str="Choose folder"):
    try:return {"path":choose_folder(title)}
    except Exception as e: raise HTTPException(500,f"Native folder dialog failed: {e}")

app.mount("/",StaticFiles(directory=str(APP_ROOT/"static"),html=True),name="static")
