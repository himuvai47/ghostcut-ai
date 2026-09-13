# Agent 1 Index Contract — Schema v4 / Agent 1 v1.3.0

`footage_index.db` is the machine source of truth. `exports/clips.jsonl` is the exchange format used by later agents.

Each record represents one semantic clip. Timing and segmentation are deterministic; Qwen never chooses timeline positions.

## Core clip record

Existing v1.2.1 semantic fields remain, including:

```text
description
primary_subject / secondary_subjects
primary_product / secondary_products
objects / people / actions
environment
shot_type
camera_motion
content_type
visual_attributes / visual_concepts
visible_text
observed_details / uncertain_inferences
quality_flags
usable / exclusion_reason
confidence_level / analysis_confidence / confidence_basis
```

Confidence remains software-derived after model validation.

## v1.3 editorial fields

```json
{
  "brand": "TAG Heuer",
  "model_name": "Carrera",
  "subject_focus": "dial",
  "shot_angle": "three_quarter",
  "editorial_role": "technical_detail",
  "visual_energy": "calm",
  "lighting_style": "dark",
  "composition": "single_subject",
  "visual_family": "carrera__technical_detail__dial__macro__three_quarter__dark__orbit",
  "visual_quality_score": 0.92,
  "editorial_usefulness_score": 0.96,
  "editorial_profile_version": 1
}
```

`brand` and `model_name` are returned only when visually supported. They are `unknown` when the evidence is insufficient.

### subject_focus

One of:

`full_product`, `dial`, `movement`, `caseback`, `case_side`, `bezel`, `bracelet`, `strap`, `clasp`, `crown_pushers`, `hands_on_product`, `wrist_wear`, `packaging`, `brand_logo`, `retail_display`, `person`, `environment`, `text_graphic`, `multiple_products`, `other`, `unknown`.

### shot_angle

One of:

`front`, `three_quarter`, `side`, `rear_caseback`, `top_down`, `low_angle`, `high_angle`, `wrist_perspective`, `mixed`, `unknown`.

### editorial_role

One of:

`product_beauty`, `technical_detail`, `movement_detail`, `wrist_lifestyle`, `handling_demo`, `retail_display`, `brand_identity`, `comparison_support`, `environment_context`, `generic_topic_broll`, `presenter`, `text_graphic`, `other`.

Obvious contradictions are normalized deterministically. For example, a `movement` focus becomes `movement_detail`; product handling becomes `handling_demo`; text/end cards become `text_graphic`.

### visual_energy

`static`, `calm`, `moderate`, `dynamic`.

### lighting_style

`bright`, `dark`, `neutral`, `high_contrast`, `natural`, `mixed`, `unknown`.

### composition

`single_subject`, `multi_subject`, `detail_only`, `person_and_product`, `text_dominant`, `environmental`, `comparison_like`, `unknown`.

## Deterministic editorial outputs

Qwen does **not** output these fields:

- `visual_family`
- `visual_quality_score`
- `editorial_usefulness_score`
- `editorial_profile_version`

Agent 1 derives them after validating the model evidence.

`visual_family` deliberately uses broad shot identity rather than raw prose. Similar shots therefore tend to share a family even when their clip IDs and timestamps differ. This gives Agent 2 a stronger signal for avoiding visually repetitive filler.

`visual_quality_score` penalizes actual quality defects such as blur, poor focus, poor exposure, obstruction, compression, and poor subject visibility.

`editorial_usefulness_score` combines quality, confidence, editorial role, identifiable product context, and subject focus. It is a B-roll selection aid, not a narration-match score.

## Face metadata

Agent 1 v1.3 preserves the v1.2.1 face layer (`FACE_SCAN_VERSION = 2`):

```text
face_status
face_scan_version
face_scan_mode
face_evidence
```

`face_status` is `no_face`, `face_visible`, or `uncertain`.

## Backward-compatible upgrade

A v1.2.1 index using the known legacy signature can be upgraded without re-segmentation. `-EditorialBackfill` reuses cached representative frames and updates only semantic analysis records to schema v4. Face metadata is preserved.

An individual failed editorial re-analysis does not destroy the old indexed clip; the previous analysis remains available and can be retried later.

## Contract rules

1. Clip timing comes from deterministic software.
2. Agent 1 never assigns narration or timeline placement.
3. Model-only identity claims must be visually supported; otherwise use `unknown`.
4. Observation and inference remain separate.
5. Machine enums are normalized before persistence.
6. Non-editorial screens remain traceable but unusable.
7. Face metadata remains independent from semantic/editorial analysis.
8. Visual-family and editorial scores are deterministic software outputs.
9. Existing v1.2.1 segmentation can be preserved during the v1.3 upgrade.
10. Agent 2 may consume these fields as additional ranking/diversity signals but remains responsible for editorial matching.
