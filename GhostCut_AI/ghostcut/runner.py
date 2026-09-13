from __future__ import annotations
import os, re, subprocess, threading, uuid
from pathlib import Path
from typing import Callable
from .libraries import face_metadata_status
from .models import Job, Project, PIPELINE_STAGES
from .projects import ProjectStore
from .settings import GhostCutSettings
from .utils import utc_now

STAGES=list(PIPELINE_STAGES)
RUN_MODES={"full","agent1","agent2","agent3","from_agent2"}

class JobRegistry:
    def __init__(self):
        self.jobs: dict[str,Job]={}; self.processes: dict[str,subprocess.Popen]={}; self.threads: dict[str,threading.Thread]={}; self.lock=threading.Lock(); self.active_job_id=""
    def get(self,jid:str)->Job:
        if jid not in self.jobs: raise KeyError(jid)
        return self.jobs[jid]
    def create(self,pid:str,run_mode:str="full",stage_states:dict[str,str]|None=None,log_path:str="")->Job:
        with self.lock:
            if self.active_job_id:
                j=self.jobs.get(self.active_job_id)
                if j and j.status in {"queued","running"}: raise RuntimeError("Another GhostCut job is already running. Let it finish or cancel it first.")
            j=Job(id=f"job_{uuid.uuid4().hex[:10]}",project_id=pid,stage_states=dict(stage_states or {s:"pending" for s in STAGES}),run_mode=run_mode,log_path=log_path)
            self.jobs[j.id]=j; self.active_job_id=j.id; return j
    def register_thread(self,jid:str,thread:threading.Thread):
        with self.lock:
            self.threads[jid]=thread

    def is_live(self,jid:str)->bool:
        with self.lock:
            proc=self.processes.get(jid)
            if proc is not None and proc.poll() is None:
                return True
            thread=self.threads.get(jid)
            # A registered-but-not-yet-started thread is reserved/live too.
            if thread is not None:
                return thread.ident is None or thread.is_alive()
            return False

    def finish_slot(self,jid:str):
        with self.lock:
            self.threads.pop(jid,None)
            self.processes.pop(jid,None)
            if self.active_job_id==jid:self.active_job_id=""

class PipelineRunner:
    def __init__(self,settings:GhostCutSettings,projects:ProjectStore,registry:JobRegistry):
        self.s=settings; self.projects=projects; self.registry=registry

    def _prepare_run(self,p:Project,mode:str):
        if mode not in RUN_MODES: raise ValueError(f"Unknown run mode: {mode}")
        if mode=="agent1" and p.library_mode!="new" and not p.footage_path:
            raise ValueError("This project uses an existing library and has no original footage folder saved, so Agent 1 cannot be fully re-indexed from this project. Re-run Agent 2/3, or create a project from New Footage to re-index.")
        states={s:p.stage_states.get(s,"pending") for s in STAGES}
        states["preflight"]="pending"
        if mode=="full":
            states.update({"index":"pending","face":"pending","match":"pending","render":"pending","done":"pending"})
        elif mode=="agent1":
            states.update({"index":"pending","face":"pending","match":"stale","render":"stale","done":"stale"})
        elif mode in {"agent2","from_agent2"}:
            states["match"]="pending"; states["render"]="pending" if mode=="from_agent2" else "stale"; states["done"]="pending" if mode=="from_agent2" else "stale"
        elif mode=="agent3":
            states["render"]="pending"; states["done"]="pending"
        p.stage_states=states; p.status="queued"; p.error=""; p.last_run_mode=mode; p.last_job_status="queued"
        p.pipeline_stage="queued"; p.pipeline_stage_label="Queued"; p.pipeline_progress=0; p.pipeline_message="Waiting to start"
        return states

    def start(self,project:Project,mode:str="full")->Job:
        states=self._prepare_run(project,mode)
        log_path=str((Path(project.root)/"logs"/"ghostcut_pipeline.log").resolve())
        job=self.registry.create(project.id,mode,states,log_path)
        thread=threading.Thread(target=self._run,args=(project,job,mode),daemon=True,name=f"ghostcut-{job.id}")
        self.registry.register_thread(job.id,thread)
        project.last_job_id=job.id
        try:
            self.projects.save(project)
        except Exception:
            # Never leave a phantom active slot when the job could not actually start.
            job.status="failed"; job.stage_label="Could not start"; job.message="GhostCut could not persist the queued job state."; job.finished_at=utc_now()
            self.registry.finish_slot(job.id)
            raise
        self._log(job,f"\n=== GhostCut job {job.id} | mode={mode} | {utc_now()} ===")
        thread.start()
        return job

    def cancel(self,job:Job):
        job.cancel_requested=True; job.message="Cancel requested…"; job.updated_at=utc_now()
        p=self.registry.processes.get(job.id)
        if p and p.poll() is None:
            try:
                if os.name=="nt": subprocess.run(["taskkill","/PID",str(p.pid),"/T","/F"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                else: p.terminate()
            except Exception: pass

    def _sync(self,p:Project,job:Job):
        p.stage_states=dict(job.stage_states); p.pipeline_stage=job.stage; p.pipeline_stage_label=job.stage_label; p.pipeline_progress=job.progress; p.pipeline_message=job.message; p.last_job_status=job.status; p.last_run_mode=job.run_mode
        self.projects.save(p)

    def _set(self,p:Project,job:Job,stage:str,label:str,progress:int,message:str,state:str="running"):
        job.stage=stage; job.stage_label=label; job.progress=max(0,min(100,int(progress))); job.message=message; job.status="running"; job.updated_at=utc_now(); job.stage_states[stage]=state; self._sync(p,job)

    def _log(self,job:Job,line:str):
        line=line.rstrip()
        if not line:return
        job.logs.append(line); job.logs[:]=job.logs[-800:]; job.updated_at=utc_now()
        if job.log_path:
            try:
                lp=Path(job.log_path); lp.parent.mkdir(parents=True,exist_ok=True)
                with lp.open("a",encoding="utf-8",errors="replace") as f: f.write(line+"\n")
            except Exception: pass

    def _powershell(self,job:Job,script:Path,args:list[str],progress_cb:Callable[[str],None]|None=None):
        cmd=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(script),*args] if os.name=="nt" else ["pwsh","-NoProfile","-File",str(script),*args]
        creationflags=getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0) if os.name=="nt" else 0
        self._log(job,"$ "+" ".join(cmd))
        p=subprocess.Popen(cmd,cwd=str(script.parent),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1,creationflags=creationflags)
        self.registry.processes[job.id]=p
        assert p.stdout is not None
        for line in p.stdout:
            self._log(job,line)
            if progress_cb: progress_cb(line)
            if job.cancel_requested:
                self.cancel(job); break
        code=p.wait(); self.registry.processes.pop(job.id,None)
        if job.cancel_requested: raise InterruptedError("Job canceled")
        if code!=0: raise RuntimeError(f"{script.name} exited with code {code}")

    def _ensure_face(self,p:Project,job:Job,a1:Path,progress:int=36):
        face=face_metadata_status(Path(p.library_path))
        if not face.get("ready"):
            self._set(p,job,"face","Updating face metadata",progress,"One-time Agent 1 face metadata upgrade is required")
            self._powershell(job,a1,["-Workspace",p.library_path,"-FaceBackfill"])
            face=face_metadata_status(Path(p.library_path))
            if not face.get("ready"): raise RuntimeError("Agent 1 face metadata is still not ready after FaceBackfill")
            job.stage_states["face"]="complete"; self._sync(p,job)
        else:
            job.stage_states["face"]="complete"; self._sync(p,job)

    def _run_agent1(self,p:Project,job:Job,a1:Path):
        self._set(p,job,"index","Indexing footage",8,"Agent 1 is building/updating the semantic footage library")
        def a1prog(line):
            if "clip" in line.lower(): job.progress=min(35,job.progress+1)
        if p.footage_path:
            self._powershell(job,a1,["-InputPath",p.footage_path,"-Workspace",p.library_path],a1prog)
            job.stage_states["index"]="complete"; self._sync(p,job)
        else:
            job.stage_states["index"]="skipped"; self._sync(p,job)
        self._ensure_face(p,job,a1)

    def _run_agent2(self,p:Project,job:Job,a1:Path,a2:Path):
        # Agent 2 requires persistent face metadata. Repair only that prerequisite when needed.
        self._ensure_face(p,job,a1,36)
        self._set(p,job,"match","Matching visuals",40,"Agent 2 is analyzing narration and selecting footage")
        total=0; matched=0
        def a2prog(line):
            nonlocal total,matched
            m=re.search(r"Visual segments:\s*(\d+)",line)
            if m: total=max(1,int(m.group(1)))
            if "INFO | Matching vs_" in line:
                matched+=1
                if total: job.progress=40+round(38*min(1,matched/total))
                job.message=f"Matching visual segment {matched}"+(f" of {total}" if total else "")
        self._powershell(job,a2,["-ScriptPath",p.script_path,"-AudioPath",p.audio_path,"-Agent1Workspace",p.library_path,"-Workspace",p.agent2_workspace],a2prog)
        job.stage_states["match"]="complete"; self._sync(p,job)
        timeline=Path(p.agent2_workspace)/"timeline.json"
        if not timeline.exists(): raise FileNotFoundError("Agent 2 completed but timeline.json was not produced")
        return timeline

    def _run_agent3(self,p:Project,job:Job,a3:Path):
        timeline=Path(p.agent2_workspace)/"timeline.json"
        if not timeline.exists(): raise FileNotFoundError("Agent 3 cannot run because Agent 2 timeline.json is missing. Re-run Agent 2 first.")
        self._set(p,job,"render","Rendering rough cut",80,"Agent 3 is rendering the selected timeline")
        def a3prog(line):
            m=re.search(r"Rendering clip\s+(\d+)\s*/\s*(\d+)",line,re.I)
            if m:
                cur,totalc=int(m.group(1)),max(1,int(m.group(2))); job.progress=80+round(17*cur/totalc); job.message=f"Rendering clip {cur} of {totalc}"
        self._powershell(job,a3,["-TimelinePath",str(timeline),"-Workspace",p.agent3_workspace],a3prog)
        rough=Path(p.agent3_workspace)/"rough_cut.mp4"
        if not rough.exists(): raise FileNotFoundError("Agent 3 completed but rough_cut.mp4 was not produced")
        job.stage_states["render"]="complete"; p.rough_cut_path=str(rough.resolve()); self._sync(p,job)
        return rough

    def _run(self,p:Project,job:Job,mode:str="full"):
        try:
            job.started_at=utc_now(); p.status="running"; self.projects.save(p)
            self._set(p,job,"preflight","Preparing project",3,"Checking project inputs and agent workspaces")
            for path,label in [(Path(p.script_path),"script"),(Path(p.audio_path),"voiceover")]:
                if not path.is_file(): raise FileNotFoundError(f"Project {label} missing: {path}")
            a1=Path(self.s.agent1_root)/"RUN_AGENT1.ps1"; a2=Path(self.s.agent2_root)/"RUN_AGENT2.ps1"; a3=Path(self.s.agent3_root)/"RUN_AGENT3.ps1"
            needed=[a1] if mode=="agent1" else ([a2,a1] if mode=="agent2" else ([a3] if mode=="agent3" else ([a2,a3,a1] if mode=="from_agent2" else [a1,a2,a3])))
            for x in needed:
                if not x.exists(): raise FileNotFoundError(f"Agent runner missing: {x}")
            job.stage_states["preflight"]="complete"; self._sync(p,job)

            if mode=="full":
                if p.library_mode=="new": self._run_agent1(p,job,a1)
                else:
                    job.stage_states["index"]="skipped"; self._sync(p,job); self._ensure_face(p,job,a1)
                self._run_agent2(p,job,a1,a2); self._run_agent3(p,job,a3)
            elif mode=="agent1":
                self._run_agent1(p,job,a1)
            elif mode=="agent2":
                self._run_agent2(p,job,a1,a2)
            elif mode=="from_agent2":
                self._run_agent2(p,job,a1,a2); self._run_agent3(p,job,a3)
            elif mode=="agent3":
                self._run_agent3(p,job,a3)

            job.status="completed"; job.finished_at=utc_now(); job.updated_at=utc_now()
            if mode in {"full","from_agent2","agent3"}:
                job.stage_states["done"]="complete"; job.stage="done"; job.stage_label="Rough cut ready"; job.progress=100; job.message="GhostCut AI finished successfully"; p.status="completed"; p.error=""
            elif mode=="agent2":
                job.stage="match"; job.stage_label="Agent 2 complete"; job.progress=78; job.message="Timeline updated. Agent 3 is stale and needs re-rendering."; p.status="stale"; p.error=""
            else:
                job.stage="face"; job.stage_label="Agent 1 complete"; job.progress=36; job.message="Footage library updated. Agent 2 and Agent 3 are stale."; p.status="stale"; p.error=""
            self._sync(p,job)
        except InterruptedError as e:
            if job.stage in job.stage_states: job.stage_states[job.stage]="canceled"
            job.status="canceled"; job.stage_label="Canceled"; job.message=str(e); job.error=str(e); job.finished_at=utc_now(); p.status="canceled"; p.error=str(e); self._sync(p,job)
        except Exception as e:
            self._log(job,f"ERROR | {e}")
            if job.stage in job.stage_states: job.stage_states[job.stage]="failed"
            job.status="failed"; job.stage_label="Pipeline failed"; job.message=str(e); job.error=str(e); job.finished_at=utc_now(); p.status="failed"; p.error=str(e); self._sync(p,job)
        finally:
            job.updated_at=utc_now(); self.registry.finish_slot(job.id)

    def reconcile_project(self,p:Project)->Project:
        """Recover projects persisted as active when no GhostCut worker still exists.

        Browser/server restarts and older state-save failures can leave ``running`` or
        ``queued`` in project.json even though there is no thread/process capable of
        completing that job.  Treat that as interrupted, preserve existing artifacts,
        and unlock stage reruns.
        """
        active_statuses={"queued","running"}
        persisted_active=(p.status in active_statuses or p.last_job_status in active_statuses)
        if not persisted_active:
            return p
        job=None
        if p.last_job_id:
            try: job=self.registry.get(p.last_job_id)
            except KeyError: job=None
        if job is not None and job.status in active_statuses and self.registry.is_live(job.id):
            return p
        if job is not None and job.status in active_statuses:
            job.status="interrupted"; job.stage_label="Interrupted"
            job.message="Previous GhostCut worker is no longer running. Choose a stage to rerun."
            job.error=""; job.finished_at=utc_now(); job.updated_at=utc_now()
            self.registry.finish_slot(job.id)
        # Preserve completed/stale stage outputs; only an explicitly running stage is stale.
        states=dict(p.stage_states or {})
        if p.pipeline_stage in states and states.get(p.pipeline_stage)=="running":
            states[p.pipeline_stage]="stale" if p.pipeline_stage in {"index","match","render","done"} else "pending"
        p.stage_states=states
        p.status="interrupted"; p.last_job_status="interrupted"
        p.pipeline_stage_label="Interrupted"
        p.pipeline_message="Previous GhostCut job is no longer running. Choose a stage to rerun."
        p.error=""
        self.projects.save(p)
        return p

    def project_snapshot(self,p:Project)->dict:
        p=self.reconcile_project(p)
        if p.last_job_id:
            try:return self.registry.get(p.last_job_id).to_dict()
            except KeyError: pass
        logs=[]; lp=Path(p.root)/"logs"/"ghostcut_pipeline.log"
        if lp.is_file():
            try: logs=lp.read_text(encoding="utf-8",errors="replace").splitlines()[-500:]
            except Exception: pass
        if not logs:
            # v1.0.0 compatibility: surface the newest agent log if GhostCut did not persist its own yet.
            candidates=[]
            for root in (Path(p.agent2_workspace)/"logs",Path(p.agent3_workspace)/"logs",Path(p.library_path)/"logs"):
                if root.is_dir(): candidates.extend(root.glob("*.log"))
            if candidates:
                latest=max(candidates,key=lambda x:x.stat().st_mtime)
                try: logs=[f"[Recovered from {latest}]",*latest.read_text(encoding="utf-8",errors="replace").splitlines()[-499:]]
                except Exception: pass
        return {"id":p.last_job_id or "snapshot","project_id":p.id,"status":p.last_job_status or p.status,"stage":p.pipeline_stage,"stage_label":p.pipeline_stage_label,"progress":p.pipeline_progress,"message":p.pipeline_message,"logs":logs,"started_at":"","updated_at":p.updated_at,"finished_at":"","error":p.error,"cancel_requested":False,"stage_states":dict(p.stage_states),"run_mode":p.last_run_mode or "full","log_path":str(lp)}
