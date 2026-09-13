# FacelessRC — Agent 1 v1.3.0: Editorial Vision Index

Agent 1 builds the persistent local footage index used by GhostCut/Agent 2. v1.3 keeps the proven v1.2.1 segmentation and face-screening behavior, but upgrades each semantic clip with **editorially useful shot metadata**.

The goal is not longer prose. The goal is to help the editor distinguish visually similar filler shots and choose stronger B-roll.

## What v1.3 adds

Each clip can now contain:

- `brand` — visible/strongly supported brand, otherwise `unknown`
- `model_name` — exact model when visually supported
- `subject_focus` — what the shot emphasizes, such as `dial`, `movement`, `bracelet`, `wrist_wear`, `full_product`
- `shot_angle` — `front`, `three_quarter`, `side`, `rear_caseback`, etc.
- `editorial_role` — generic B-roll use such as `product_beauty`, `technical_detail`, `movement_detail`, `wrist_lifestyle`, `handling_demo`, `retail_display`
- `visual_energy` — `static`, `calm`, `moderate`, `dynamic`
- `lighting_style` — `bright`, `dark`, `neutral`, `high_contrast`, `natural`, etc.
- `composition` — broad visual arrangement
- `visual_family` — deterministic family signature used to group shots that are editorially similar
- `visual_quality_score` — deterministic 0–1 technical/visual quality score
- `editorial_usefulness_score` — deterministic 0–1 B-roll usefulness score

Qwen identifies the observable shot attributes. Agent 1 software calculates `visual_family`, `visual_quality_score`, and `editorial_usefulness_score` deterministically so those values remain consistent across the library.

## Existing v1.2.1 workspace: use Editorial Backfill

Do **not** delete or rebuild your existing footage index.

After applying the v1.3 patch, upgrade the existing clips with:

```powershell
cd "D:\AI\FacelessRC\Agent1_Indexing"

.\RUN_AGENT1.ps1 `
  -Workspace "D:\AI\FacelessRC\indexes\rich-people-stopped-buying-these-15-watch-brands-library" `
  -EditorialBackfill
```

This pass:

1. keeps all existing physical shots and semantic clip boundaries,
2. reuses the representative-frame cache,
3. calls local `qwen2.5vl:7b` only for clips needing the v1.3 editorial schema,
4. preserves existing face metadata,
5. preserves the old semantic record if an individual upgrade fails,
6. refreshes `exports\clips.jsonl` and QC.

If a cached representative frame is missing, only that clip's representative frames are regenerated. Source videos are not re-segmented.

The backfill is resumable. Successfully upgraded clips are skipped on the next run. Use `-Force` only when you intentionally want to re-run every clip.

For a short smoke test before upgrading all 570 clips:

```powershell
.\RUN_AGENT1.ps1 `
  -Workspace "D:\AI\FacelessRC\indexes\rich-people-stopped-buying-these-15-watch-brands-library" `
  -EditorialBackfill `
  -Limit 10
```

Then run the same command without `-Limit` for the full library.

## New footage

Normal indexing automatically uses the v1.3 editorial schema:

```powershell
.\RUN_AGENT1.ps1 `
  -InputPath "D:\AI\FacelessRC\Footage" `
  -Workspace "D:\AI\FacelessRC\indexes\my_library"
```

If Agent 1 sees a structurally compatible v1.2.1 index during a normal run, v1.3 recognizes the legacy signature and preserves the existing segmentation instead of unnecessarily rebuilding it.

## Face metadata remains unchanged

v1.3 preserves Agent 1 v1.2.1 face metadata and `FACE_SCAN_VERSION = 2`:

- `no_face`
- `face_visible`
- `uncertain`

Use `-FaceBackfill` only for face metadata. Use `-EditorialBackfill` for the v1.3 editorial metadata.

## Pipeline

```text
source video
  -> deterministic physical shots
  -> semantic clips
  -> representative frames
  -> Qwen2.5-VL visual analysis
  -> v1.3 editorial shot metadata
  -> deterministic quality/usefulness scoring
  -> deterministic visual-family signature
  -> persistent face metadata
  -> SQLite + JSONL
  -> QC
```

Agent 1 still does **not** choose narration matches or timeline positions. That remains Agent 2's job.

## Outputs

```text
<workspace>\
  footage_index.db
  exports\clips.jsonl
  frames\...
  logs\...
  qc_report.json
  last_run_summary.json
  last_face_backfill_summary.json
  last_editorial_backfill_summary.json
```

## Inspect

```powershell
.\.venv\Scripts\python.exe inspect_index.py "D:\AI\FacelessRC\indexes\rich-people-stopped-buying-these-15-watch-brands-library"
```

The inspector shows brand/model, editorial role, subject focus, quality/usefulness scores, visual family, and face status.

## Requirements

- Windows 11
- Python 3.11+
- FFmpeg + FFprobe
- Ollama
- `qwen2.5vl:7b`
- local CUDA/GPU acceleration recommended

No paid APIs or cloud processing are used.
