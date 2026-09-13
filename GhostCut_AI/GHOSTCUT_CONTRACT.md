# GhostCut AI v1.0.1 Contract

GhostCut AI is the local UI/orchestration layer for the FacelessRC pipeline.

## Boundaries
GhostCut does not make editorial footage decisions. Agent 1 indexes footage, Agent 2 chooses footage, and Agent 3 renders the timeline. GhostCut manages projects, validates paths, runs/resumes those agents, streams progress/logs, and surfaces results.

## Persistent project pipeline
Every project owns a permanent Pipeline View. It remains available when the project is running, completed, stale, failed, or canceled. Pipeline stage state is persisted in `project.json`, and GhostCut appends pipeline logs to `logs/ghostcut_pipeline.log`, so reopening the app does not erase the last visible pipeline state.

## Stage controls
GhostCut supports these explicit run modes:
- **Agent 1 only** — refresh the footage index / face metadata. Agent 2 and Agent 3 become stale.
- **Agent 2 only** — rebuild `timeline.json`. Agent 3 becomes stale.
- **Agent 3 only** — render the current timeline.
- **Run From Agent 2** — Agent 2, then Agent 3.
- **Run Entire Pipeline** — normal autonomous project flow.

Existing-library projects that do not store an original footage folder cannot fully re-index Agent 1 from the project panel; the UI disables that action and explains why. Face metadata may still be repaired automatically as an Agent 2 prerequisite.

## Artifact access
The Pipeline View exposes available stage outputs directly: the Agent 1 library, Agent 2 `timeline.json`, and Agent 3 rough cut. Project folders and existing rough cuts remain openable even if a newer upstream stage has made the render stale.

## Autonomous flow
1. Create a project with script + voiceover.
2. Select an existing Agent 1 index or a new footage folder.
3. For new footage, Agent 1 indexes it. Existing indexes skip normal indexing.
4. If persistent Agent 1 face metadata is missing/stale, GhostCut runs `-FaceBackfill` once.
5. Agent 2 creates `timeline.json`.
6. Agent 3 creates `rough_cut.mp4`.
7. GhostCut preserves the project pipeline for later inspection or selective reruns.

## Safety and concurrency
Only one production job runs at a time. This avoids competing Ollama/GPU workloads. Cancel requests terminate the active child process tree on Windows. Stage reruns deliberately invalidate downstream state before execution so GhostCut never presents an old render as current after an upstream change.

## Local-only
The UI binds to `127.0.0.1` by default. GhostCut does not upload project media. Ollama, FFmpeg, and all three agents remain local.


## v1.0.2 Windows state-save hotfix
- Project state writes are serialized inside `ProjectStore`.
- JSON writes use a unique temporary file per writer instead of a shared `project.json.tmp`.
- Atomic replacement retries short Windows `PermissionError` / sharing violations before failing.
- Temporary files are cleaned up after success or failure.

## v1.0.3 Orphan-job recovery
A persisted `queued` or `running` status is authoritative only while the current GhostCut server still owns a live registered worker thread/process for that job. If the server restarts, a previous start fails before its worker launches, or the worker otherwise disappears, GhostCut changes the persisted project/job snapshot to `interrupted`, preserves existing artifacts and stage outputs, and unlocks stage reruns. Browser polling also recovers from a missing in-memory job by reopening the persistent pipeline snapshot.
