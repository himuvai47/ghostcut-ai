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

# timeline.json — Phase 3 Contract

Agent 2 owns editorial decisions. Phase 3 must execute this file deterministically.

Top-level fields include `schema_version`, `project`, `audio`, `policy`, `segments[]`, and `editorial_review[]`.

Each selected clip includes Agent 1 `clip_id`, source path/video, source and timeline ranges, reason, match level, deterministic confidence, use number, Agent 1 `face_status`, and optional `coverage_entity` for multi-entity narration.

Face status rules:
- `face_visible` must never appear in a selected clip.
- `no_face` is allowed.
- `uncertain` is allowed when Agent 2's configured policy permits it (default: allowed).
- the face status comes from Agent 1 persistent metadata; Phase 3 must not reinterpret it.

Renderer rules:
- never extend source ranges beyond Agent 2's approved ranges;
- preserve the supplied voiceover as timing backbone;
- render explicit gap/placeholder policy for `no_match` segments;
- do not silently fill a `coverage_shortfall` with unapproved footage.

Multi-entity rule:
- `segments[].required_entities` may contain multiple named entities.
- Separate selected clips may each carry `coverage_entity`; together they satisfy the segment.
- Phase 3 renders them in the supplied timeline order and does not require one clip to contain all entities.


Editorial tolerance (v1.0.7+):
- a selected clip may be broadly relevant without proving every fine narration detail;
- missing soft attributes are not renderer/QC failures;
- Phase 3 should execute the supplied timeline as-is rather than attempting to improve semantic precision.


Pacing contract (v1.0.8):
- normal selected clips must be at least 3.0 seconds and at most 8.0 seconds on the timeline;
- Phase 3 must preserve these Agent 2 cut lengths exactly;
- do not subdivide a legal Agent 2 cut into shorter flash cuts;
- longer narration units may contain multiple sequential 3–8 second cuts.


## Gap-fill metadata — v1.0.9

A selected clip may include `gap_filler: true` when it was chosen only to cover a remaining legal 3+ second gap with generally related B-roll. The timeline policy records the gap-fill behavior. Renderer behavior is unchanged: render these clips exactly like normal selected clips.


## v1.0.13 — Beat-specific long-form semantics

- Paced child beats no longer inherit one broad parent visual request.
- Every split beat derives its retrieval target from that beat's own narration.
- Hard named entities are kept only when explicitly present in the child narration.
- Generic roles/concepts such as buyer, presenter, community, graphics, overlays and certification text are not hard entities for faceless B-roll.
- Weak-but-related B-roll no longer triggers global editorial rematching.
- Global review is reserved for actual clip reuse and receives only the risky repeated segments.
- Long-form pacing/diversity rules from v1.0.11 remain unchanged.


### Coverage completion
`coverage_shortfall` should be zero for a matched segment whenever a legal >=3 second remainder can be covered by unused face-safe library footage. v1.0.13 performs a deterministic completion pass after the two-candidate broad reranker before allowing such a gap to remain.

## v1.0.14 — Visual diversity / near-duplicate avoidance

Agent 2 now treats visual repetition as more than exact clip-ID reuse. It uses existing Agent-1 metadata and cached retrieval embeddings to penalize and, when enough alternatives exist, strongly deprioritize:

- clips from the same source video used in the immediately preceding beats,
- clips from the same nearby source-time region (default 20 seconds),
- near-identical semantic/framing matches across separate source files, and
- overused source videos when other relevant sources remain available.

No additional vision-model or Qwen call is added for diversity. Named strict/high-specificity requirements keep relevance priority; the diversity layer is strongest for normal faceless B-roll.

The run summary and timeline now expose `source_distribution`, `same_source_consecutive_cuts`, `nearby_source_region_reuses`, `semantic_near_duplicate_reuses`, and `visual_diversity_score` so visual repetition can be audited directly instead of relying only on exact `repeated_clips`.


## v1.0.15 — Editorial freedom metadata

Each timeline segment may include `match_mode`, `context_anchor`, `planned_requested_visual`, `requested_visual`, `planned_required_entities`, and effective `required_entities`. Selected filler clips may include `editorial_filler: true`. Agent 3 does not reinterpret these fields; it renders the approved clip ranges exactly as supplied.

The three modes mean:
- `literal`: exact visible subject/detail matters when available; context-aware filler is allowed only as fallback.
- `thematic`: broad topic/brand relevance is sufficient.
- `brand_filler`: varied same-brand/same-topic footage is the intended editorial choice, not a failed match.
