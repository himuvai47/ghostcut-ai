# Agent 3 Contract — v1.0.0

Agent 3 is deterministic. It does not choose footage, call AI models, or reinterpret narration.

## Inputs
- Agent 2 `timeline.json`
- source footage paths referenced by the timeline
- narration audio path referenced by `timeline.audio.source`

## Output
- `rough_cut.mp4`
- `render_plan.json`
- `qc_report.json`
- `run_summary.json`

## Rendering policy
- 1920×1080, 30 fps by default
- center crop after aspect-preserving scale
- source audio removed
- narration is the only audio
- H.264 NVENC when available, libx264 fallback
- no transitions/effects/captions/music/SFX/grading
- timeline gaps are freeze-filled from an adjacent normalized frame
- final duration must match narration within 0.10 seconds
- selected source windows are cached by source metadata + timing + render settings
