from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect a FacelessRC Agent 1 footage index")
    p.add_argument("workspace", nargs="?", default="./index")
    p.add_argument("--clip", help="Print one exact clip UID")
    p.add_argument("--limit", type=int, default=15)
    args = p.parse_args()

    db_path = Path(args.workspace) / "footage_index.db"
    if not db_path.exists():
        print(f"Index not found: {db_path}")
        return 2
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.clip:
        row = conn.execute(
            """SELECT c.*, s.shot_uid, src.relative_path AS source_video
               FROM clips c JOIN shots s ON s.id=c.shot_id JOIN sources src ON src.id=c.source_id
               WHERE c.clip_uid=?""", (args.clip,)
        ).fetchone()
        if not row:
            print("Clip not found")
            return 1
        out = dict(row)
        for key in ("representative_times_json", "representative_paths_json", "analysis_json", "face_evidence_json"):
            if out.get(key):
                try:
                    out[key] = json.loads(out[key])
                except json.JSONDecodeError:
                    pass
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    sources = conn.execute("SELECT relative_path,status,duration FROM sources ORDER BY relative_path").fetchall()
    print(f"Sources: {len(sources)}")
    for s in sources:
        print(f"  {s['status']:<9} {s['duration']:8.2f}s  {s['relative_path']}")
    print("\nIndexed clips:")
    rows = conn.execute(
        """SELECT c.clip_uid,c.start_time,c.end_time,c.description,c.analysis_json,c.face_status,src.relative_path
           FROM clips c JOIN sources src ON src.id=c.source_id
           WHERE c.status='INDEXED' ORDER BY src.relative_path,c.clip_index LIMIT ?""", (args.limit,)
    ).fetchall()
    for r in rows:
        analysis = json.loads(r["analysis_json"] or "{}")
        usable = analysis.get("usable", True)
        content = analysis.get("content_type", "unknown")
        product = analysis.get("primary_product", "none")
        exclusion = analysis.get("exclusion_reason", "none")
        print(f"  {r['clip_uid']}  {r['start_time']:.2f}-{r['end_time']:.2f}  {r['relative_path']}")
        role = analysis.get("editorial_role", "legacy")
        focus = analysis.get("subject_focus", "legacy")
        family = analysis.get("visual_family", "legacy")
        brand = analysis.get("brand", "unknown")
        model = analysis.get("model_name", "unknown")
        quality = analysis.get("visual_quality_score", "legacy")
        usefulness = analysis.get("editorial_usefulness_score", "legacy")
        print(f"    type={content} usable={usable} product={product} exclusion={exclusion} face={r['face_status'] or 'not_scanned'}")
        print(f"    brand={brand} model={model} role={role} focus={focus} quality={quality} usefulness={usefulness}")
        print(f"    family={family}")
        print(f"    {r['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
