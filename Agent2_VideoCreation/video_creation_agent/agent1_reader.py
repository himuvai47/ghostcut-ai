from __future__ import annotations
import hashlib, json, sqlite3
from pathlib import Path
from .models import IndexedClip

FACE_STATUSES={"no_face","face_visible","uncertain"}
MIN_FACE_SCAN_VERSION=2


def _clean_meta_text(value, default="unknown"):
    text=" ".join(str(value or "").split()).strip()
    if not text or text.casefold() in {"none","unknown","n/a","null"}:
        return default
    return text

def _score(value, default=0.5):
    try:
        return max(0.0,min(1.0,float(value)))
    except (TypeError,ValueError):
        return float(default)

def _editorial_version(analysis):
    try:
        return max(0,int(analysis.get("editorial_profile_version") or 0))
    except (TypeError,ValueError):
        return 0

def _face_columns(conn: sqlite3.Connection) -> set[str]:
    return {str(r[1]) for r in conn.execute("PRAGMA table_info(clips)").fetchall()}

def load_agent1_clips(workspace: Path) -> list[IndexedClip]:
    dbp = workspace / "footage_index.db"
    if not dbp.exists(): raise FileNotFoundError(f"Agent 1 DB not found: {dbp}")
    conn = sqlite3.connect(f"file:{dbp.as_posix()}?mode=ro", uri=True); conn.row_factory=sqlite3.Row
    try:
        cols=_face_columns(conn)
        required={"face_status","face_scan_version"}
        missing=sorted(required-cols)
        if missing:
            raise RuntimeError(
                "Agent 1 index is missing persistent face metadata ("+", ".join(missing)+"). "
                "Upgrade Agent 1 to v1.2.1+ and run -FaceBackfill once for this workspace."
            )
        rows = conn.execute("""SELECT c.*, s.shot_uid, src.source_uid, src.relative_path, src.absolute_path, src.fingerprint AS source_fingerprint
        FROM clips c JOIN shots s ON s.id=c.shot_id JOIN sources src ON src.id=c.source_id
        WHERE c.status='INDEXED' ORDER BY src.relative_path,c.clip_index""").fetchall()
        bad=[]; out=[]
        keys=cols
        for r in rows:
            status=(r['face_status'] or '').strip()
            version=r['face_scan_version']
            if status not in FACE_STATUSES or version is None or int(version)<MIN_FACE_SCAN_VERSION:
                bad.append(r['clip_uid']); continue
            a=json.loads(r['analysis_json'] or '{}'); rep=json.loads(r['representative_paths_json'] or '[]'); times=json.loads(r['representative_times_json'] or '[]')
            raw=json.dumps({'analysis':a,'source_fingerprint':r['source_fingerprint'],'start_time':r['start_time'],'end_time':r['end_time'],'representative_times':times},sort_keys=True,ensure_ascii=False)
            out.append(IndexedClip(
                r['clip_uid'],r['source_uid'],r['relative_path'],r['absolute_path'],float(r['start_time']),float(r['end_time']),float(r['duration']),
                a.get('description',r['description'] or ''),a.get('primary_subject','unknown'),a.get('primary_product','none'),a.get('secondary_products',[]),a.get('objects',[]),a.get('people',[]),a.get('actions',[]),a.get('environment','unknown'),a.get('shot_type','unknown'),a.get('camera_motion','unknown'),a.get('content_type','other'),a.get('visual_attributes',[]),a.get('visual_concepts',[]),a.get('visible_text',[]),a.get('observed_details',[]),a.get('uncertain_inferences',[]),bool(a.get('usable',r['usable'] if r['usable'] is not None else True)),float(a.get('analysis_confidence',r['analysis_confidence'] or .75)),rep,times,hashlib.sha256(raw.encode()).hexdigest(),
                status,int(version),str(r['face_scan_mode'] or '') if 'face_scan_mode' in keys else '',
                _clean_meta_text(a.get('brand')),
                _clean_meta_text(a.get('model_name')),
                _clean_meta_text(a.get('subject_focus')),
                _clean_meta_text(a.get('shot_angle')),
                _clean_meta_text(a.get('editorial_role'),'other'),
                _clean_meta_text(a.get('visual_energy'),'calm'),
                _clean_meta_text(a.get('lighting_style')),
                _clean_meta_text(a.get('composition')),
                _clean_meta_text(a.get('visual_family')),
                _score(a.get('visual_quality_score')),
                _score(a.get('editorial_usefulness_score')),
                _editorial_version(a),
            ))
        if bad:
            sample=', '.join(bad[:5])
            more=f" (+{len(bad)-5} more)" if len(bad)>5 else ''
            raise RuntimeError(
                f"Agent 1 face metadata is missing or stale for {len(bad)} indexed clip(s): {sample}{more}. "
                "Run Agent 1 v1.2.1+ with -FaceBackfill, then rerun Agent 2."
            )
        return out
    finally: conn.close()
