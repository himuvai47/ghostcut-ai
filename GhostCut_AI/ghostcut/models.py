from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
from .utils import utc_now

PIPELINE_STAGES=("preflight","index","face","match","render","done")

@dataclass(slots=True)
class Project:
    id: str
    name: str
    slug: str
    root: str
    script_path: str
    audio_path: str
    library_mode: str
    library_path: str = ""
    footage_path: str = ""
    status: str = "created"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    last_job_id: str = ""
    rough_cut_path: str = ""
    error: str = ""
    agent2_workspace: str = ""
    agent3_workspace: str = ""
    stage_states: dict[str,str] = field(default_factory=dict)
    pipeline_stage: str = "created"
    pipeline_stage_label: str = "Ready"
    pipeline_progress: int = 0
    pipeline_message: str = "Project created"
    last_run_mode: str = ""
    last_job_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(slots=True)
class Job:
    id: str
    project_id: str
    status: str = "queued"
    stage: str = "queued"
    stage_label: str = "Queued"
    progress: int = 0
    message: str = "Waiting to start"
    logs: list[str] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = field(default_factory=utc_now)
    finished_at: str = ""
    error: str = ""
    cancel_requested: bool = False
    stage_states: dict[str, str] = field(default_factory=dict)
    run_mode: str = "full"
    log_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        d=asdict(self)
        d["logs"] = self.logs[-500:]
        return d
