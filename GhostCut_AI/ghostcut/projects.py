from __future__ import annotations
import shutil, threading, uuid
from pathlib import Path
from .models import Project, PIPELINE_STAGES
from .settings import GhostCutSettings
from .utils import read_json, safe_copy, slugify, utc_now, write_json

class ProjectStore:
    def __init__(self, settings: GhostCutSettings):
        self.settings=settings
        self.root=Path(settings.projects_root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Stage updates can arrive from the request thread and the background runner
        # within milliseconds of one another. Serialize writes to project.json so
        # project state cannot race on Windows.
        self._save_lock=threading.RLock()

    def _project_file(self, project_root: Path) -> Path:
        return project_root / "project.json"

    def _normalize(self,p:Project)->Project:
        states=dict(p.stage_states or {})
        for s in PIPELINE_STAGES: states.setdefault(s,"pending")
        # Upgrade v1.0.0 projects without losing already-produced work.
        timeline=Path(p.agent2_workspace)/"timeline.json" if p.agent2_workspace else Path("__missing__")
        rough=Path(p.rough_cut_path) if p.rough_cut_path else (Path(p.agent3_workspace)/"rough_cut.mp4" if p.agent3_workspace else Path("__missing__"))
        if p.status=="completed" or rough.is_file():
            states.update({"preflight":"complete","index":"complete" if p.library_mode=="new" else "skipped","face":"complete","match":"complete","render":"complete","done":"complete"})
            if rough.is_file() and not p.rough_cut_path: p.rough_cut_path=str(rough.resolve())
            if p.pipeline_stage in {"", "created"}: p.pipeline_stage="done"
            if p.pipeline_stage_label in {"", "Ready"}: p.pipeline_stage_label="Rough cut ready"
            if p.pipeline_progress==0: p.pipeline_progress=100
            if p.pipeline_message in {"", "Project created"}: p.pipeline_message="GhostCut AI finished successfully"
            if not p.last_job_status: p.last_job_status="completed"
        else:
            if timeline.is_file(): states["match"]="complete"
            if rough.is_file(): states["render"]="complete"
        p.stage_states=states
        return p

    def create(self, name: str, script_path: str, audio_path: str, library_mode: str, library_path: str = "", footage_path: str = "") -> Project:
        if library_mode not in {"existing","new"}: raise ValueError("library_mode must be 'existing' or 'new'")
        sp, ap = Path(script_path), Path(audio_path)
        if not sp.is_file(): raise FileNotFoundError(f"Script not found: {sp}")
        if not ap.is_file(): raise FileNotFoundError(f"Voiceover not found: {ap}")
        if library_mode=="existing" and not Path(library_path).is_dir(): raise FileNotFoundError(f"Index workspace not found: {library_path}")
        if library_mode=="new" and not Path(footage_path).exists(): raise FileNotFoundError(f"Footage path not found: {footage_path}")
        slug=slugify(name)
        pid=f"gc_{uuid.uuid4().hex[:10]}"
        project_root=self.root/f"{slug}_{pid[-6:]}"
        inputs=project_root/"inputs"; inputs.mkdir(parents=True, exist_ok=True)
        script_copy=safe_copy(sp, inputs/("script"+sp.suffix.lower()))
        audio_copy=safe_copy(ap, inputs/("voiceover"+ap.suffix.lower()))
        agent2=project_root/"agent2_output"; agent3=project_root/"agent3_output"
        agent2.mkdir(); agent3.mkdir(); (project_root/"logs").mkdir()
        if library_mode=="new":
            lib=Path(self.settings.indexes_root)/f"{slug}-library"
            library_path=str(lib.resolve())
        p=Project(id=pid,name=name.strip() or "Untitled Project",slug=slug,root=str(project_root.resolve()),script_path=str(script_copy.resolve()),audio_path=str(audio_copy.resolve()),library_mode=library_mode,library_path=str(Path(library_path).resolve()),footage_path=str(Path(footage_path).resolve()) if footage_path else "",agent2_workspace=str(agent2.resolve()),agent3_workspace=str(agent3.resolve()),stage_states={s:"pending" for s in PIPELINE_STAGES})
        self.save(p)
        return p

    def save(self, p: Project) -> None:
        with self._save_lock:
            p.updated_at=utc_now()
            write_json(self._project_file(Path(p.root)),p.to_dict())

    def get(self, project_id: str) -> Project:
        for p in self.list():
            if p.id==project_id: return p
        raise KeyError(project_id)

    def list(self) -> list[Project]:
        self.root.mkdir(parents=True,exist_ok=True)
        out=[]
        for f in self.root.glob("*/project.json"):
            try: out.append(self._normalize(Project(**read_json(f))))
            except Exception: continue
        return sorted(out,key=lambda x:x.updated_at,reverse=True)

    def delete(self, project_id: str) -> None:
        p=self.get(project_id); shutil.rmtree(p.root)
