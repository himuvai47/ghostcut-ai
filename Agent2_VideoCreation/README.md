Agent 2 v1.0.15 — Editorial Freedom / Filler Mode

- Every visual beat now carries `match_mode`: `literal`, `thematic`, or `brand_filler`.
- `literal` is intentionally sparse and reserved for exact models/products/details/processes/comparisons where exact footage materially helps.
- `thematic` accepts broad same-topic / same-brand footage; abstract sentence details do not need literal depiction.
- `brand_filler` intentionally uses attractive varied same-brand / same-topic footage and skips the Qwen candidate reranker for that beat.
- Context anchors carry the main brand/topic through abstract narration without letting a one-off model permanently hijack the next beats.
- If literal/thematic matching cannot cover a beat, the existing gap fallback is now context-aware and relaxes toward same-brand/topic footage.
- v1.0.14 near-duplicate avoidance, source rotation, 3–8s pacing, unused-first selection, and face metadata rules remain active.
- New `editorial_plan.json` records planned vs effective visual requirements and mode counts.

This is deliberately designed for commentary-heavy faceless videos where exact sentence-to-footage matching is often impossible or unnecessary.

Agent 2 v1.0.14 — Beat-Specific Long-form Matching

- Long semantic ideas are deterministically split into ~8s visual beats (10s ceiling).
- Exact matching is attempted first; if unavailable, unused topical B-roll is acceptable.
- Named entities are relaxed only in the fallback path, so missing Nomos/METAS/etc. cannot freeze the edit.
- Repeats are blocked while at least 12 legal unused clips remain.
- Matched segments with meaningful uncovered timeline spans now fail QC instead of being silently freeze-filled by Agent 3.
- Existing 3–8s clip pacing, Agent 1 face metadata, long-form word-ID repair, and fast reranking remain intact.

Agent 2 v1.0.10 long-form planner reliability: malformed/nonexistent Qwen word IDs are repaired deterministically without an extra model call; valid gaps/overlaps remain errors.

# FacelessRC Agent 2 — Video Creation Agent v1.0.9

Agent 2 converts a script + real voiceover timing + the Agent 1 footage index into an auditable `timeline.json`. It makes editorial decisions but does not render video.


## What changed in v1.0.9

- Coverage gaps of **3 seconds or more** trigger one cheap broad B-roll fallback instead of being left black.
- The fallback reuses the existing local retrieval index; it does **not** restart exhaustive retrieval.
- At most **2** generally related candidates are sent to the fast no-thinking reranker for each gap-fill attempt.
- Unused related footage is preferred; repeat footage is considered only after unused fallback options are exhausted.
- Strict/high-specificity named products remain strict during gap fill.
- Existing **3–8 second** pacing and longer-clip preference remain unchanged.

## What changed in v1.0.8

- Normal selected source clips are paced at **3.0–8.0 seconds**.
- Clips shorter than 3 seconds are excluded from normal matching.
- Among candidates with the same match level, Agent 2 prefers the clip that can stay on screen longer.
- If an 8-second cut would leave a tiny remainder under 3 seconds, deterministic pacing rebalances the cuts instead of creating a flash. Example: a 9.5-second visual beat becomes 6.5s + 3.0s, not 8.0s + 1.5s.
- Planner output shorter than 3 seconds is merged with a neighboring visual unit when possible.
- QC now validates the 3–8 second cut-duration contract.

## What changed in v1.0.7

- Matching is **relevance-first**, not literal-detail-first. Correct subject/product + broadly related B-roll is normally enough.
- Fine details such as color, hour markers, texture, finishing, exact component motion, framing and camera movement are soft preferences and cannot cause a hard reject by themselves.
- Candidate reranking uses Qwen3.5 with **thinking disabled** and starts with only the best 2 candidates. It widens only when necessary.
- The global editorial review is now **risk-only**: it runs only when a weak match or repeated clip is actually selected. Normal strong/acceptable unique edits skip the extra Qwen pass.
- Multi-product narration can still use separate sequential clips for each named product.

## What changed in v1.0.6

- Visual planning is semantic-first: `requested_visual` describes **what must be visible**, not camera/framing technique.
- Close-up/macro/static/pan/zoom/split-view language is stripped from hard visual requirements; framing remains a soft preference.
- Multi-entity narration can be covered collectively by separate sequential clips. Example: one G-Shock clip followed by one Apple Watch clip.
- Each selected clip can carry `coverage_entity` so the global reviewer knows which named entity it satisfies.
- Global review judges all clips in a segment collectively and no longer expects one clip to contain every named product in a comparison.

## What changed in v1.0.5

- Agent 2 no longer extracts frames or runs Qwen2.5-VL for face screening.
- Face status is read directly from Agent 1 v1.2.1+ persistent clip metadata.
- `face_visible` is rejected. `no_face` and, by default, `uncertain` are allowed.
- Face-visible clips are removed before embedding/retrieval.
- Qwen3.5 reranking is progressive: judge the best 4 first and widen only if needed.
- Old Agent 2 face-cache files may remain on disk, but they are no longer consulted.

## Required Agent 1 index

Agent 2 v1.0.8 requires Agent 1 face metadata version 2 or newer. Existing older indexes must be upgraded once with Agent 1 v1.2.1+:

```powershell
.\RUN_AGENT1.ps1 `
  -Workspace "D:\AI\FacelessRC\indexes\multivideo_test" `
  -FaceBackfill
```

New footage indexed by Agent 1 v1.2.1+ already receives face metadata during normal indexing.

## Architecture

1. Validate script, audio, Agent 1 QC/database, face metadata, and eligible source paths.
2. Convert narration audio to 16 kHz mono WAV using FFmpeg.
3. Run local whisper.cpp and align token timestamps to the canonical script.
4. Ask local `qwen3.5:9b` to group the timed script into meaningful visual units, then deterministically remove invented filming/style constraints from hard requirements.
5. Reject Agent 1 clips marked `face_visible`; allow `no_face` and configured `uncertain` clips.
6. Build/cache local embeddings for eligible clips using `nomic-embed-text`.
7. Hybrid retrieve by semantic similarity + lexical overlap + structured metadata + Agent 1 confidence.
8. Ask `qwen3.5:9b` to judge the strongest candidates with thinking disabled. Reranking starts with the top 2 and expands only when needed.
9. For multi-entity segments, retrieve each required entity separately and cover the segment with sequential clips when useful.
10. Deterministic pacing selects longer relevant footage first and enforces 3–8 second source cuts, while avoiding sub-3-second remainders.
11. Run global review only for weak/repeated selections, then write `timeline.json`, no-match records, candidate audit files, and QC.

## Clip-use policy

`strong relevant unused` → `acceptable/close/filler unused` → `relevant second use` → `no_match`

A semantic clip can be used at most **2 times total**. A repeat cannot happen within the configured segment gap (default 2). The same clip is never selected twice inside one visual segment. On a second use, deterministic software chooses an alternate sub-range when possible.

Normal selected cuts are **minimum 3 seconds and maximum 8 seconds**. Longer relevant candidates are preferred inside the same match-quality tier. If a visual unit is longer than 8 seconds, Agent 2 splits it into multiple legal cuts rather than stretching one source clip.

## Face policy

Agent 1 owns face perception. Agent 2 only reads the saved result:

- `no_face` → allowed
- `uncertain` → allowed by default, with a small ranking preference for confirmed `no_face` when relevance is otherwise similar
- `face_visible` → hard reject

Hands, wrists, arms, torso/body, and presenters whose face is not visible are not rejected merely for containing a person.

## Models used by Agent 2

- `qwen3.5:9b` — visual planning, candidate judgment, global editorial review
- `nomic-embed-text` — semantic retrieval
- whisper.cpp `small.en` — local narration timing

**Qwen2.5-VL is no longer required by Agent 2.** Agent 1 handles footage perception and face metadata.

## Outputs

- `alignment.json`
- `visual_segments.json`
- `retrieval_candidates.jsonl`
- `match_decisions.json`
- `no_matches.json`
- `matching_progress.json`
- `timeline.json`
- `qc_report.json`
- `run_summary.json`
- `cache/retrieval_cache.db`
- `logs/`

## Resumability

Alignment and visual segmentation are reused when script/audio are unchanged. Clip embeddings are cached by Agent 1 analysis hash. Matching checkpoints after every visual segment. A change to Agent 1's database or matching policy invalidates matching without forcing Whisper or visual planning to rerun.

## Installation

```powershell
cd "D:\AI\FacelessRC\Agent2_VideoCreation"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\SETUP_WINDOWS.ps1
```

Setup checks FFmpeg/Ollama, creates the venv, ensures `qwen3.5:9b` and `nomic-embed-text` are installed, prepares whisper.cpp, then runs precheck.

## Run

```powershell
.\RUN_AGENT2.ps1 `
  -ScriptPath "D:\AI\FacelessRC\Projects\agent2_test\script.txt" `
  -AudioPath "D:\AI\FacelessRC\Projects\agent2_test\voiceover.mp3" `
  -Agent1Workspace "D:\AI\FacelessRC\indexes\multivideo_test" `
  -Workspace "D:\AI\FacelessRC\projects\agent2_test_output"
```

Do not use `-Force` for normal resume runs.

## V1 boundaries

Agent 2 does not render, transition, caption, grade, animate, change speed, add music/SFX, or fabricate missing footage. `timeline.json` is its final contract for Phase 3.


## v1.0.13 — Beat-specific long-form semantics

- Paced child beats no longer inherit one broad parent visual request.
- Every split beat derives its retrieval target from that beat's own narration.
- Hard named entities are kept only when explicitly present in the child narration.
- Generic roles/concepts such as buyer, presenter, community, graphics, overlays and certification text are not hard entities for faceless B-roll.
- Weak-but-related B-roll no longer triggers global editorial rematching.
- Global review is reserved for actual clip reuse and receives only the risky repeated segments.
- Long-form pacing/diversity rules from v1.0.11 remain unchanged.


## v1.0.13 — deterministic coverage completion

When normal matching leaves a legal 3+ second remainder, Agent 2 still asks Qwen about at most two broad fallback candidates. If those are rejected or have the wrong duration shape, Agent 2 now scans the already-ranked unused face-safe library locally and selects broadly related B-roll that can legally complete the remaining 3–8 second window. This adds no extra model call and prevents a segment from being marked matched with a large uncovered span simply because the first two fallback clips were too short.

## v1.0.14 — Visual diversity / near-duplicate avoidance

Agent 2 now treats visual repetition as more than exact clip-ID reuse. It uses existing Agent-1 metadata and cached retrieval embeddings to penalize and, when enough alternatives exist, strongly deprioritize:

- clips from the same source video used in the immediately preceding beats,
- clips from the same nearby source-time region (default 20 seconds),
- near-identical semantic/framing matches across separate source files, and
- overused source videos when other relevant sources remain available.

No additional vision-model or Qwen call is added for diversity. Named strict/high-specificity requirements keep relevance priority; the diversity layer is strongest for normal faceless B-roll.

The run summary and timeline now expose `source_distribution`, `same_source_consecutive_cuts`, `nearby_source_region_reuses`, `semantic_near_duplicate_reuses`, and `visual_diversity_score` so visual repetition can be audited directly instead of relying only on exact `repeated_clips`.

