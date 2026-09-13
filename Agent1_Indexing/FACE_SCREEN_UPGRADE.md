# Agent 1 v1.2.0 — Face Screen Upgrade

## Existing workspace command

```powershell
.\RUN_AGENT1.ps1 `
  -Workspace "D:\AI\FacelessRC\indexes\multivideo_test" `
  -FaceBackfill
```

Expected behavior for the current 65-clip test index:

```text
Agent 1 v1.2.0 starting
Face backfill: 65 indexed clip(s), 65 need scan
  face-backfill 1/65
  face-backfill 10/65
  ...
  face-backfill 65/65
```

The final summary reports counts for `no_face`, `face_visible`, and `uncertain`.

Run the same command again to verify resumability. It should report `0 need scan` and `clips_scanned: 0`.

## Policy for Agent 2

- `no_face` -> allowed
- `uncertain` -> allowed
- `face_visible` -> reject

Relevance should remain more important than preferring `no_face` over `uncertain`.
