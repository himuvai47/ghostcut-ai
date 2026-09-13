# GhostCut AI v1.0.1

GhostCut AI is the local command center for FacelessRC. It provides a premium browser UI around Agent 1 (indexing), Agent 2 (editorial matching), and Agent 3 (rendering), with no cloud processing.

## v1.0.1 — Pipeline Controls

Projects are no longer one-shot records. Every project can reopen its **Project Pipeline** at any time, including after completion, failure, cancellation, or an app restart.

New controls:
- Re-run Agent 1 only
- Re-run Agent 2 only
- Re-render Agent 3 only
- Run From Agent 2
- Run Entire Pipeline
- Open Agent 1 library, Agent 2 timeline, and Agent 3 video from the pipeline
- Reopen the pipeline from Projects or from the result screen
- Persist pipeline state and logs across GhostCut restarts

GhostCut automatically marks downstream work **stale** when an upstream stage is rerun. Example: rerunning Agent 2 leaves Agent 1 current but marks Agent 3 stale until you rerender.

## Install / update
For a fresh install, use the normal GhostCut package. For v1.0.0 installations, apply the v1.0.1 patch from its extracted patch folder with `APPLY_PATCH.ps1`.

## Launch
```powershell
cd "D:\AI\FacelessRC\GhostCut_AI"
.\RUN_GHOSTCUT.ps1
```

Open `http://127.0.0.1:8765` if the browser does not open automatically.

## Normal flow
Create a project, choose script + voiceover + footage source, and click **Create Rough Cut**. GhostCut still runs the normal autonomous Agent 1 → Agent 2 → Agent 3 flow. The new stage controls are optional and are designed for iteration/debugging.


## v1.0.2 Windows state-save hotfix
- Project state writes are serialized inside `ProjectStore`.
- JSON writes use a unique temporary file per writer instead of a shared `project.json.tmp`.
- Atomic replacement retries short Windows `PermissionError` / sharing violations before failing.
- Temporary files are cleaned up after success or failure.

## v1.0.3 — Orphan Job Recovery
- Projects persisted as `queued` / `running` are reconciled against the actual in-memory GhostCut worker.
- If no worker/process exists, the project is automatically recovered as **interrupted** and rerun controls unlock.
- A disappeared job during browser polling is recovered through the persistent Pipeline View instead of leaving the UI stuck active.
- Job threads are tracked explicitly, including the brief queued-before-start state.
- If initial project-state persistence fails while starting a job, the global active-job slot is released immediately so GhostCut cannot deadlock itself.
