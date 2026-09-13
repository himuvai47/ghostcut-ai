# GhostCut AI

Local-first autonomous faceless-video rough-cut pipeline.

## Pipeline
1. Agent 1 â€” footage indexing / face metadata / Editorial Vision metadata
2. Agent 2 â€” narration alignment, compact editorial planning, diverse footage matching, timeline generation
3. Agent 3 â€” deterministic FFmpeg rendering
4. GhostCut AI â€” local UI and orchestration

## V1 frozen versions
See `VERSION_MANIFEST.json`.

## Requirements
- Windows 11
- Python
- FFmpeg / ffprobe
- Ollama
- NVIDIA GPU recommended

Development models used by the frozen system:
- Qwen2.5-VL 7B via Ollama (vision)
- Qwen3.5 9B via Ollama (reasoning)
- nomic-embed-text (embeddings)
- local whisper.cpp small.en (timing)

No paid API or cloud processing is required by the core pipeline.

Before publishing publicly, run the release safety scan and review third-party dependency licenses.
