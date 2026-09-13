from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .models import MediaInfo, TimeRange, VisionAnalysis


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_uid TEXT NOT NULL UNIQUE,
    relative_path TEXT NOT NULL,
    absolute_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    fingerprint_mode TEXT NOT NULL,
    index_signature TEXT NOT NULL,
    duration REAL NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps REAL NOT NULL,
    codec TEXT NOT NULL,
    format_name TEXT NOT NULL,
    has_audio INTEGER NOT NULL,
    rotation INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    indexed_at TEXT
);

CREATE TABLE IF NOT EXISTS shots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shot_uid TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    shot_index INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    detection_method TEXT NOT NULL,
    UNIQUE(source_id, shot_index)
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_uid TEXT NOT NULL UNIQUE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    shot_id INTEGER NOT NULL REFERENCES shots(id) ON DELETE CASCADE,
    clip_index INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    representative_times_json TEXT NOT NULL DEFAULT '[]',
    representative_paths_json TEXT NOT NULL DEFAULT '[]',
    analysis_json TEXT,
    description TEXT,
    usable INTEGER,
    analysis_confidence REAL,
    status TEXT NOT NULL,
    error TEXT,
    model TEXT,
    prompt_version TEXT,
    analysis_version INTEGER,
    face_status TEXT,
    face_scan_version INTEGER,
    face_scan_mode TEXT,
    face_evidence_json TEXT,
    face_scanned_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(source_id, clip_index)
);

CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_clips_source_status ON clips(source_id, status);
CREATE INDEX IF NOT EXISTS idx_clips_usable ON clips(usable);
"""


class IndexDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate_schema()
        self.conn.commit()


    def _migrate_schema(self) -> None:
        """Add v1.2 face metadata columns to existing v1.1.x indexes in place."""
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(clips)")}
        additions = {
            "face_status": "TEXT",
            "face_scan_version": "INTEGER",
            "face_scan_mode": "TEXT",
            "face_evidence_json": "TEXT",
            "face_scanned_at": "TEXT",
        }
        for name, sql_type in additions.items():
            if name not in columns:
                self.conn.execute(f"ALTER TABLE clips ADD COLUMN {name} {sql_type}")

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.conn.execute("BEGIN")
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def get_source(self, source_uid: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM sources WHERE source_uid=?", (source_uid,)).fetchone()

    def upsert_source(
        self,
        *, source_uid: str, relative_path: str, absolute_path: str,
        file_size: int, mtime_ns: int, fingerprint: str, fingerprint_mode: str,
        index_signature: str, media: MediaInfo, status: str,
    ) -> int:
        now = utc_now()
        existing = self.get_source(source_uid)
        if existing is None:
            cur = self.conn.execute(
                """INSERT INTO sources(
                    source_uid,relative_path,absolute_path,file_size,mtime_ns,fingerprint,fingerprint_mode,index_signature,
                    duration,width,height,fps,codec,format_name,has_audio,rotation,status,error,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)""",
                (source_uid, relative_path, absolute_path, file_size, mtime_ns, fingerprint, fingerprint_mode,
                 index_signature, media.duration, media.width, media.height, media.fps, media.codec, media.format_name,
                 int(media.has_audio), media.rotation, status, now, now),
            )
            self.conn.commit()
            return int(cur.lastrowid)
        self.conn.execute(
            """UPDATE sources SET relative_path=?, absolute_path=?, file_size=?, mtime_ns=?, fingerprint=?,
               fingerprint_mode=?, index_signature=?, duration=?, width=?, height=?, fps=?, codec=?, format_name=?,
               has_audio=?, rotation=?, status=?, error=NULL, updated_at=? WHERE id=?""",
            (relative_path, absolute_path, file_size, mtime_ns, fingerprint, fingerprint_mode, index_signature,
             media.duration, media.width, media.height, media.fps, media.codec, media.format_name,
             int(media.has_audio), media.rotation, status, now, existing["id"]),
        )
        self.conn.commit()
        return int(existing["id"])

    def reset_source_children(self, source_id: int) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM clips WHERE source_id=?", (source_id,))
            conn.execute("DELETE FROM shots WHERE source_id=?", (source_id,))

    def set_source_status(self, source_id: int, status: str, error: str | None = None, indexed: bool = False) -> None:
        now = utc_now()
        self.conn.execute(
            "UPDATE sources SET status=?, error=?, updated_at=?, indexed_at=CASE WHEN ? THEN ? ELSE indexed_at END WHERE id=?",
            (status, error, now, int(indexed), now, source_id),
        )
        self.conn.commit()

    def add_shot(self, source_id: int, shot_uid: str, shot_index: int, rng: TimeRange, method: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO shots(shot_uid,source_id,shot_index,start_time,end_time,duration,detection_method) VALUES(?,?,?,?,?,?,?)",
            (shot_uid, source_id, shot_index, rng.start, rng.end, rng.duration, method),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_clip(self, source_id: int, shot_id: int, clip_uid: str, clip_index: int, rng: TimeRange) -> int:
        cur = self.conn.execute(
            """INSERT INTO clips(clip_uid,source_id,shot_id,clip_index,start_time,end_time,duration,status,updated_at)
               VALUES(?,?,?,?,?,?,?,'PENDING',?)""",
            (clip_uid, source_id, shot_id, clip_index, rng.start, rng.end, rng.duration, utc_now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def clips_for_source(self, source_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT c.*, s.shot_uid, s.shot_index FROM clips c JOIN shots s ON s.id=c.shot_id WHERE c.source_id=? ORDER BY c.clip_index",
            (source_id,),
        ))

    def pending_clips_for_source(self, source_id: int) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            """SELECT c.*, s.shot_uid, s.shot_index FROM clips c JOIN shots s ON s.id=c.shot_id
               WHERE c.source_id=? AND c.status!='INDEXED' ORDER BY c.clip_index""",
            (source_id,),
        ))

    def save_frame_evidence(self, clip_id: int, times: list[float], paths: list[Path]) -> None:
        self.conn.execute(
            "UPDATE clips SET representative_times_json=?, representative_paths_json=?, status='FRAMES_READY', error=NULL, updated_at=? WHERE id=?",
            (json.dumps(times), json.dumps([str(p) for p in paths]), utc_now(), clip_id),
        )
        self.conn.commit()

    def save_analysis(self, clip_id: int, analysis: VisionAnalysis, model: str, prompt_version: str, analysis_version: int) -> None:
        payload = analysis.to_dict()
        self.conn.execute(
            """UPDATE clips SET analysis_json=?, description=?, usable=?, analysis_confidence=?, status='INDEXED', error=NULL,
               model=?, prompt_version=?, analysis_version=?, updated_at=? WHERE id=?""",
            (json.dumps(payload, ensure_ascii=False), analysis.description, int(analysis.usable), analysis.analysis_confidence,
             model, prompt_version, analysis_version, utc_now(), clip_id),
        )
        self.conn.commit()

    def fail_clip(self, clip_id: int, status: str, error: str) -> None:
        self.conn.execute(
            "UPDATE clips SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error[:4000], utc_now(), clip_id),
        )
        self.conn.commit()

    def source_counts(self, source_id: int) -> dict[str, int]:
        shot_count = self.conn.execute("SELECT COUNT(*) FROM shots WHERE source_id=?", (source_id,)).fetchone()[0]
        clip_count = self.conn.execute("SELECT COUNT(*) FROM clips WHERE source_id=?", (source_id,)).fetchone()[0]
        indexed = self.conn.execute("SELECT COUNT(*) FROM clips WHERE source_id=? AND status='INDEXED'", (source_id,)).fetchone()[0]
        failed = clip_count - indexed
        return {"shots": shot_count, "clips": clip_count, "indexed": indexed, "failed": failed}

    def all_indexed_clips(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            """SELECT c.*, s.shot_uid, s.shot_index, src.source_uid, src.relative_path AS source_video,
                      src.duration AS source_duration, src.width AS source_width, src.height AS source_height,
                      src.fps AS source_fps, src.codec AS source_codec, src.absolute_path AS source_absolute_path
               FROM clips c
               JOIN shots s ON s.id=c.shot_id
               JOIN sources src ON src.id=c.source_id
               WHERE c.status='INDEXED'
               ORDER BY src.relative_path, c.clip_index"""
        ))

    def save_face_scan(
        self, clip_id: int, *, status: str, version: int, mode: str, evidence: dict[str, Any]
    ) -> None:
        self.conn.execute(
            """UPDATE clips SET face_status=?, face_scan_version=?, face_scan_mode=?, face_evidence_json=?,
               face_scanned_at=?, updated_at=? WHERE id=?""",
            (status, version, mode, json.dumps(evidence, ensure_ascii=False), utc_now(), utc_now(), clip_id),
        )
        self.conn.commit()

    def indexed_clips_needing_face_scan(self, version: int, source_id: int | None = None, force: bool = False) -> list[sqlite3.Row]:
        where = ["c.status='INDEXED'"]
        params: list[Any] = []
        if source_id is not None:
            where.append("c.source_id=?")
            params.append(source_id)
        if not force:
            where.append("(c.face_scan_version IS NULL OR c.face_scan_version!=? OR c.face_status IS NULL)")
            params.append(version)
        return list(self.conn.execute(
            f"""SELECT c.*, s.shot_uid, s.shot_index, src.source_uid, src.relative_path AS source_video,
                       src.absolute_path AS source_absolute_path
                FROM clips c
                JOIN shots s ON s.id=c.shot_id
                JOIN sources src ON src.id=c.source_id
                WHERE {' AND '.join(where)}
                ORDER BY src.relative_path, c.clip_index""",
            tuple(params),
        ))


    def indexed_clips_needing_editorial(
        self, schema_version: int, prompt_version: str, *, source_id: int | None = None, force: bool = False
    ) -> list[sqlite3.Row]:
        where = ["c.status='INDEXED'"]
        params: list[Any] = []
        if source_id is not None:
            where.append("c.source_id=?")
            params.append(source_id)
        if not force:
            where.append("(c.analysis_version IS NULL OR c.analysis_version<? OR c.prompt_version IS NULL OR c.prompt_version!=?)")
            params.extend([schema_version, prompt_version])
        return list(self.conn.execute(
            f"""SELECT c.*, s.shot_uid, s.shot_index, src.source_uid, src.relative_path AS source_video,
                       src.absolute_path AS source_absolute_path
                FROM clips c
                JOIN shots s ON s.id=c.shot_id
                JOIN sources src ON src.id=c.source_id
                WHERE {' AND '.join(where)}
                ORDER BY src.relative_path, c.clip_index""",
            tuple(params),
        ))

    def source_editorial_complete(self, source_id: int, schema_version: int, prompt_version: str) -> bool:
        total = self.conn.execute(
            "SELECT COUNT(*) FROM clips WHERE source_id=? AND status='INDEXED'", (source_id,)
        ).fetchone()[0]
        current = self.conn.execute(
            """SELECT COUNT(*) FROM clips WHERE source_id=? AND status='INDEXED'
               AND analysis_version>=? AND prompt_version=?""",
            (source_id, schema_version, prompt_version),
        ).fetchone()[0]
        return total > 0 and total == current

    def update_source_index_signature(self, source_id: int, index_signature: str) -> None:
        self.conn.execute(
            "UPDATE sources SET index_signature=?, updated_at=? WHERE id=?",
            (index_signature, utc_now(), source_id),
        )
        self.conn.commit()

    def all_sources(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM sources ORDER BY relative_path"))
