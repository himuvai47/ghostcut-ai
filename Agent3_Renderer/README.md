# FacelessRC Agent 3 Renderer v1.0.0

Agent 3 turns Agent 2's `timeline.json` into a real rough-cut MP4. It contains no AI model calls.

## Quick start
```powershell
cd "D:\AI\FacelessRC\Agent3_Renderer"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\SETUP_WINDOWS.ps1
.\CHECK_AGENT3.ps1
.\RUN_AGENT3.ps1 `
  -TimelinePath "D:\AI\FacelessRC\projects\agent2_test_output\timeline.json" `
  -Workspace "D:\AI\FacelessRC\projects\agent3_test_output"
```

## What it does
1. Validates timeline, narration, source files, FFmpeg/FFprobe.
2. Converts timeline boundaries to deterministic 30fps frame ranges.
3. Trims each source window, scales/crops to 1920×1080, removes source audio, and caches it.
4. Freeze-fills any unexpected timeline gap.
5. Concatenates normalized intermediates.
6. Adds the original narration only.
7. Runs ffprobe QC and fails the run if output duration/resolution/audio/video are invalid.

## Cache
Second runs reuse unchanged normalized clips. Use `-Force` to rebuild the intermediates.

## V1 exclusions
No transitions, music, SFX, captions, motion graphics, grading, speed ramps, zooms, or AI reframing.
