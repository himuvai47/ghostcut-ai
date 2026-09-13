from __future__ import annotations

import json
from pathlib import Path

from .db import IndexDB


def export_jsonl(db: IndexDB, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for row in db.all_indexed_clips():
            analysis = json.loads(row["analysis_json"] or "{}")
            record = {
                "clip_id": row["clip_uid"],
                "source_id": row["source_uid"],
                "source_video": row["source_video"],
                "parent_shot_id": row["shot_uid"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "duration": row["duration"],
                "representative_frames": json.loads(row["representative_times_json"] or "[]"),
                "face_status": row["face_status"] or "unknown",
                "face_scan_version": row["face_scan_version"],
                "face_scan_mode": row["face_scan_mode"],
                "face_evidence": json.loads(row["face_evidence_json"] or "{}"),
                **analysis,
                "model": row["model"],
                "prompt_version": row["prompt_version"],
                "analysis_version": row["analysis_version"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
