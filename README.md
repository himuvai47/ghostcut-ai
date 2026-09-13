# GhostCut AI — Local AI Video Editor for Automated Faceless Video Production

**GhostCut AI** is a local-first AI video editing and faceless video automation system that automatically turns a **script, voiceover, and footage library into an edited rough-cut video**.

GhostCut analyzes your footage with local vision models, understands the narration, selects relevant B-roll, avoids repetitive shots and visible faces, builds an editorial timeline, and renders the final video automatically with FFmpeg.

It is designed for **faceless YouTube videos, documentary-style videos, commentary videos, educational content, product videos, video essays, and automated content production**.

> Local AI video editing. No paid APIs. No cloud processing. Your footage stays on your computer.

---

## What Is GhostCut AI?

GhostCut AI is an **autonomous AI rough-cut generator** built for creators who work with large libraries of B-roll and want to automate the repetitive parts of video editing.

Give GhostCut:

- a video script
- a finished voiceover
- a folder of source footage

GhostCut can automatically:

1. analyze and index the footage
2. detect usable B-roll
3. identify visible faces
4. understand products, brands, objects, scenes, and shot types
5. align the narration to the voiceover
6. decide where visual cuts should happen
7. match footage to each narration beat
8. use thematic or same-brand filler when literal footage is unavailable
9. reduce repetitive footage
10. generate an editorial timeline
11. render the rough cut with FFmpeg

The result is a **usable first-pass video edit that can be polished manually instead of edited from scratch**.

---

# Key Features

## AI Footage Indexing

GhostCut uses a local vision-language model to analyze source footage and build a searchable visual library.

The footage index can contain information such as:

- products and objects
- brands
- watch or product models
- actions
- environments
- camera framing
- shot type
- close-up / medium / wide shots
- technical details
- visible text and logos
- editorial role
- visual family
- subject focus
- visual quality
- editorial usefulness
- face visibility

This allows GhostCut to reuse a footage library across multiple videos without analyzing everything again.

---

## Automatic B-Roll Matching

GhostCut matches footage to narration using semantic retrieval, local embeddings, editorial reasoning, and visual-diversity rules.

It supports three editorial matching modes:

### Literal

Used when an exact visual is important.

Examples:

- a specific product
- a named watch model
- a mechanical movement
- a product comparison
- a specific feature

### Thematic

Used when broadly related visuals are enough.

Examples:

- luxury watches
- collectors
- brand reputation
- purchasing behavior
- product value

### Brand Filler

Used when narration is abstract and there is no useful literal visual.

Instead of forcing an inaccurate match, GhostCut can intentionally use attractive, unused footage from the same brand or topic.

This is especially useful for **faceless commentary videos and video essays**, where much of the narration cannot be represented literally.

---

## Visual Diversity and Repeat Avoidance

GhostCut is designed to avoid the common AI-video problem of repeatedly selecting footage that technically matches but looks almost identical.

The matching system considers:

- exact clip reuse
- nearby source-video regions
- semantic near-duplicates
- source-video rotation
- visual-family reuse
- editorial role
- subject focus
- footage quality
- editorial usefulness

This helps produce a rough cut that feels more intentionally edited instead of simply selecting the highest semantic match repeatedly.

---

## Face-Aware Faceless Video Editing

GhostCut stores persistent face metadata during footage indexing.

Clips can be classified as:

- `no_face`
- `face_visible`
- `uncertain`

For faceless-video workflows, visible-face footage can be automatically excluded before footage matching.

---

## Local AI — No Paid APIs Required

GhostCut is designed to run locally.

The current GhostCut V1 stack uses:

- **Qwen2.5-VL 7B** through Ollama for visual footage analysis
- **Qwen3.5 9B** through Ollama for editorial reasoning
- **nomic-embed-text** for semantic retrieval
- **whisper.cpp small.en** for voiceover timing and alignment
- **FFmpeg / ffprobe** for deterministic video rendering

The core workflow does not require OpenAI, Anthropic, Google, or other paid cloud AI APIs.

Your:

- source footage
- scripts
- voiceovers
- embeddings
- indexes
- timelines
- rendered videos

can remain on your own computer.

---

# GhostCut Pipeline

GhostCut is built as four separate components.

```text
Source Footage
      |
      v
+-----------------------------+
| Agent 1 — Footage Indexing  |
| Vision analysis             |
| Face metadata               |
| Editorial Vision metadata   |
+-----------------------------+
      |
      v
Searchable Footage Library
      |
      |        Script + Voiceover
      |               |
      +-------+-------+
              |
              v
+-----------------------------+
| Agent 2 — Editorial Engine  |
| Voiceover alignment         |
| Editorial beat planning     |
| B-roll retrieval            |
| Literal / thematic / filler |
| Visual diversity            |
| Timeline generation         |
+-----------------------------+
              |
              v
         timeline.json
              |
              v
+-----------------------------+
| Agent 3 — Video Renderer    |
| FFmpeg                      |
| GPU acceleration when valid |
| Voiceover assembly          |
| Final QC                    |
+-----------------------------+
              |
              v
         rough_cut.mp4
```

**GhostCut AI** sits above these agents as the local user interface and pipeline orchestrator.

---

# Agent 1 — AI Footage Indexing

Agent 1 converts raw video footage into a searchable visual index.

It performs:

- video metadata inspection
- shot detection
- semantic clip creation
- representative-frame extraction
- local vision-model analysis
- face classification
- structured footage metadata extraction
- Editorial Vision indexing
- JSONL and SQLite export

GhostCut can later search this index instead of repeatedly analyzing the original footage.

### Editorial Vision

GhostCut V1 includes richer metadata designed specifically for video editing.

Examples include:

```text
brand
model_name
subject_focus
shot_angle
editorial_role
visual_energy
lighting_style
composition
visual_family
visual_quality_score
editorial_usefulness_score
```

This helps distinguish footage that may be semantically similar but visually repetitive.

---

# Agent 2 — AI Editorial Matching

Agent 2 is the editorial decision engine.

It takes:

- a script
- a voiceover
- an Agent 1 footage index

and produces:

```text
timeline.json
```

Agent 2 handles:

- voiceover alignment
- deterministic visual-beat creation
- compact AI editorial planning
- semantic footage retrieval
- editorial match modes
- broad thematic matching
- intentional filler
- visual diversity
- source balancing
- near-duplicate avoidance
- face-safe footage selection
- timeline coverage
- quality control

GhostCut intentionally does **not** require every sentence to have a perfect literal visual.

For many faceless-video workflows, good editing means using relevant, attractive, varied B-roll while the narration carries the story.

---

# Agent 3 — Automated FFmpeg Video Rendering

Agent 3 converts the generated timeline into a rendered rough cut.

It is deterministic and does not use an AI model.

Features include:

- FFmpeg rendering
- automatic source-audio removal
- voiceover assembly
- aspect-ratio preservation
- 1920×1080 output
- 30 FPS output
- H.264 encoding
- GPU/NVENC capability detection
- CPU fallback
- intermediate clip caching
- render QC
- final duration validation

Typical output:

```text
rough_cut.mp4
```

---

# GhostCut AI Interface

GhostCut AI provides a local interface for managing the complete workflow.

It supports:

- project creation
- footage-library selection
- script selection
- voiceover selection
- pipeline execution
- live logs
- progress monitoring
- individual Agent reruns
- persistent stage status
- stale-stage detection
- project recovery
- output preview
- opening generated files and folders

You can rerun:

```text
Agent 1 only
Agent 2 only
Agent 3 only
Run From Agent 2
Full Pipeline
```

without rebuilding the entire project.

---

# Designed for Faceless YouTube Automation

GhostCut is particularly suited to content such as:

- faceless YouTube videos
- YouTube automation channels
- video essays
- luxury-product commentary
- watch videos
- finance commentary
- documentary videos
- educational videos
- history videos
- business videos
- tech videos
- product analysis
- review videos
- narrated list videos
- voiceover-driven content

It is designed around a practical reality of faceless editing:

**not every sentence can or should have a perfectly literal visual.**

GhostCut therefore prioritizes a balance between:

```text
relevance
visual quality
visual variety
brand/topic consistency
editorial pacing
```

---

# GhostCut V1 Stable

The current frozen V1 stack is:

| Component | Version | Purpose |
|---|---:|---|
| GhostCut AI | 1.0.3 | Local UI and orchestration |
| Agent 1 | 1.3.0 | Footage indexing and Editorial Vision |
| Agent 2 | 1.0.20 | Editorial planning and footage matching |
| Agent 3 | 1.0.0 | FFmpeg video rendering |

See:

```text
VERSION_MANIFEST.json
```

for the frozen release manifest.

---

# Requirements

GhostCut V1 currently targets Windows.

Recommended environment:

- Windows 11
- Python
- FFmpeg
- ffprobe
- Ollama
- whisper.cpp
- NVIDIA GPU recommended
- sufficient local storage for footage and model files

Development hardware used during GhostCut V1 testing included an NVIDIA RTX-class GPU.

---

# Local AI Models

The frozen GhostCut V1 development configuration uses:

### Vision

```text
qwen2.5vl:7b
```

Used for footage understanding and Editorial Vision indexing.

### Editorial Reasoning

```text
qwen3.5:9b
```

Used for narration interpretation and editorial planning.

### Embeddings

```text
nomic-embed-text
```

Used for semantic footage retrieval.

### Voiceover Alignment

```text
whisper.cpp
small.en
```

Used locally for speech timing and narration alignment.

---

# Example Workflow

```text
1. Add footage
       ↓
2. Agent 1 indexes the footage
       ↓
3. Add script + voiceover
       ↓
4. Agent 2 creates visual beats
       ↓
5. GhostCut matches relevant B-roll
       ↓
6. Agent 2 creates timeline.json
       ↓
7. Agent 3 renders the edit
       ↓
8. Review rough_cut.mp4
       ↓
9. Optional manual polish in Premiere Pro / DaVinci Resolve
```

The goal is not to completely replace a professional editor.

The goal is to eliminate a large amount of repetitive first-pass editing work.

---

# Why GhostCut?

Traditional faceless-video production can involve hours of:

- searching through footage
- finding B-roll
- checking clips
- avoiding duplicate visuals
- aligning cuts to narration
- assembling the timeline
- rendering previews

GhostCut automates much of that process locally.

Instead of starting from an empty timeline, the creator starts from an AI-generated rough cut.

---

# Privacy and Local Processing

GhostCut is designed around local processing.

When configured with local Ollama models, whisper.cpp, embeddings, and FFmpeg, your media does not need to be uploaded to an external AI service.

This can be useful for:

- private footage
- unreleased videos
- client projects
- proprietary media libraries
- creators who prefer offline/local AI workflows

---

# Project Status

**GhostCut V1.0.0 is the first frozen stable development release.**

The current system is capable of producing usable automated rough cuts from real long-form narration and indexed footage libraries.

Development will continue on future versions while V1 remains preserved as a stable baseline.

---

# Roadmap

Potential future improvements include:

- stronger Editorial Vision metadata
- improved visual-family clustering
- smarter shot-sequence composition
- better footage-quality ranking
- more advanced source balancing
- automatic music integration
- motion graphics
- captions
- transitions
- stock-footage integration
- automatic media acquisition
- improved UI
- portable Windows build
- desktop application launcher
- Windows installer
- broader hardware support

---

# Technology Stack

GhostCut combines:

- Python
- Ollama
- Qwen vision-language models
- local LLM reasoning
- semantic embeddings
- whisper.cpp
- FFmpeg
- SQLite
- JSON / JSONL
- local web UI
- NVIDIA GPU acceleration

---

# Search Keywords

GhostCut AI is related to:

**AI video editor, local AI video editor, automatic video editor, automated video editing, faceless video automation, YouTube automation software, automatic B-roll matching, AI B-roll generator, AI footage matching, autonomous video editor, AI rough cut generator, local LLM video editor, Ollama video editing, Qwen video analysis, AI video production, faceless YouTube video editor, automatic footage indexing, semantic video search, local video AI, FFmpeg automation, AI video editing pipeline, automatic rough cut, video essay automation, content creation automation, B-roll automation.**

---

# Disclaimer

GhostCut is an experimental local AI video-production system under active development.

AI-generated footage matches and editorial decisions should be reviewed before final publication.

Third-party models, libraries, FFmpeg, Ollama, whisper.cpp, and other dependencies remain subject to their respective licenses.

---

# GhostCut AI

**Turn a script, voiceover, and footage library into an automatically edited faceless-video rough cut — locally.**
