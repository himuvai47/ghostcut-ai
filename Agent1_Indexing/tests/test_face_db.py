import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from indexing_agent.db import IndexDB
from indexing_agent.models import MediaInfo, TimeRange, VisionAnalysis


class FaceDatabaseTests(unittest.TestCase):
    def test_existing_old_clips_table_migrates_in_place(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "old.db"
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE clips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clip_uid TEXT NOT NULL UNIQUE,
                    source_id INTEGER NOT NULL,
                    shot_id INTEGER NOT NULL,
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
                    updated_at TEXT NOT NULL
                );
            """)
            conn.commit(); conn.close()
            db = IndexDB(db_path)
            cols = {r[1] for r in db.conn.execute("PRAGMA table_info(clips)")}
            db.close()
            for name in {"face_status", "face_scan_version", "face_scan_mode", "face_evidence_json", "face_scanned_at"}:
                self.assertIn(name, cols)

    def test_face_scan_saved_without_changing_analysis(self):
        with tempfile.TemporaryDirectory() as td:
            db = IndexDB(Path(td) / "x.db")
            media = MediaInfo(duration=10, width=100, height=100, fps=30, codec="h264")
            sid = db.upsert_source(source_uid="s", relative_path="a.mp4", absolute_path="C:/a.mp4", file_size=1,
                                   mtime_ns=1, fingerprint="x", fingerprint_mode="sampled", index_signature="sig",
                                   media=media, status="PROBED")
            shot = db.add_shot(sid, "sh", 1, TimeRange(0, 10), "test")
            clip = db.add_clip(sid, shot, "cl", 1, TimeRange(0, 5))
            analysis = VisionAnalysis(description="watch detail")
            db.save_analysis(clip, analysis, "m", "p", 3)
            before = db.conn.execute("SELECT analysis_json,status FROM clips WHERE id=?", (clip,)).fetchone()
            db.save_face_scan(clip, status="uncertain", version=1, mode="test", evidence={"reason":"blur", "frames":[]})
            after = db.conn.execute("SELECT analysis_json,status,face_status,face_scan_version FROM clips WHERE id=?", (clip,)).fetchone()
            self.assertEqual(before["analysis_json"], after["analysis_json"])
            self.assertEqual(after["status"], "INDEXED")
            self.assertEqual(after["face_status"], "uncertain")
            self.assertEqual(after["face_scan_version"], 1)
            db.close()
