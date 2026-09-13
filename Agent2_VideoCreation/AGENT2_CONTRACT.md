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

# Agent 2 Contract — v1.0.9

**Input:** canonical script, voiceover file, passing Agent 1 v1.2.1+ workspace with face metadata version 2+.

**Editorial AI:** `qwen3.5:9b` for visual planning, candidate judgment, global review.

**Perception source:** Agent 1. Agent 2 performs no runtime face/image analysis.

**Retrieval:** `nomic-embed-text` local embeddings plus deterministic lexical/metadata scoring.

**Timing:** local whisper.cpp token timing aligned deterministically to the user script.

**Output:** `timeline.json` where every visual segment is `matched` or `no_match`.

Hard invariants:
- only Agent 1 indexed clip IDs may be selected;
- only Agent 1 `usable=true` clips may be selected;
- `face_visible` clips are rejected;
- `no_face` and configured `uncertain` clips may be selected;
- clip use count <= 2;
- second use is considered only after reasonable unused candidates are exhausted/insufficient;
- source ranges stay inside Agent 1 semantic clip boundaries;
- AI never manipulates files or executes cuts;
- missing footage is represented explicitly as `no_match`.

Planning/matching invariants added in v1.0.6:
- semantic subject/entity/detail requirements outrank cinematography;
- framing/camera motion is soft unless the narration itself explicitly makes it essential;
- multiple required named entities may be satisfied collectively by separate sequential clips;
- global review evaluates a segment's selected clips collectively, not as if every clip must show every entity.


Relevance/speed invariants added in v1.0.7:
- ordinary faceless B-roll does not need perfect one-to-one coverage of every spoken detail;
- fine attributes (color, hour markers, texture, finishing, exact component motion, framing/camera motion) are soft preferences;
- hard failure is reserved for clearly wrong required named entities/products or plainly unrelated main subjects;
- reranking begins with two candidates and uses no-thinking structured Qwen calls;
- global review runs only when a selected clip is weak or repeated.


Pacing invariants added in v1.0.8:
- normal selected timeline clips are >= 3.0s and <= 8.0s;
- source clips shorter than 3.0s are not normal selection candidates;
- within the same match level, longer usable clips are preferred;
- tiny under-3s tail cuts are prevented by deterministic rebalancing;
- visual units shorter than 3s are merged with a neighbor when possible;
- QC validates the pacing contract.


## Coverage-gap fallback — v1.0.9

If a matched narration segment still has at least 3.0 seconds uncovered after normal matching, Agent 2 must try a cheap generally-related B-roll fallback before leaving a gap. It reuses the existing retrieval ranking, evaluates at most two fallback candidates with the fast no-thinking reranker, prefers unused footage, then respects the normal max-two repeat policy. Fine attributes may be relaxed; strict/high named entities are not relaxed away. Cuts must still obey the 3–8 second pacing contract.


## v1.0.13 — Beat-specific long-form semantics

- Paced child beats no longer inherit one broad parent visual request.
- Every split beat derives its retrieval target from that beat's own narration.
- Hard named entities are kept only when explicitly present in the child narration.
- Generic roles/concepts such as buyer, presenter, community, graphics, overlays and certification text are not hard entities for faceless B-roll.
- Weak-but-related B-roll no longer triggers global editorial rematching.
- Global review is reserved for actual clip reuse and receives only the risky repeated segments.
- Long-form pacing/diversity rules from v1.0.11 remain unchanged.


### Coverage completion (v1.0.13)
A matched segment must not retain a >=3 second uncovered remainder when legal unused face-safe library footage exists. The model-assisted broad fallback remains capped at two candidates; after that, deterministic ranked-library coverage completion may select broadly related unused B-roll without another Qwen call. All selected cuts must still obey the configured 3–8 second pacing rules.

## v1.0.14 — Visual diversity / near-duplicate avoidance

Agent 2 now treats visual repetition as more than exact clip-ID reuse. It uses existing Agent-1 metadata and cached retrieval embeddings to penalize and, when enough alternatives exist, strongly deprioritize:

- clips from the same source video used in the immediately preceding beats,
- clips from the same nearby source-time region (default 20 seconds),
- near-identical semantic/framing matches across separate source files, and
- overused source videos when other relevant sources remain available.

No additional vision-model or Qwen call is added for diversity. Named strict/high-specificity requirements keep relevance priority; the diversity layer is strongest for normal faceless B-roll.

The run summary and timeline now expose `source_distribution`, `same_source_consecutive_cuts`, `nearby_source_region_reuses`, `semantic_near_duplicate_reuses`, and `visual_diversity_score` so visual repetition can be audited directly instead of relying only on exact `repeated_clips`.

