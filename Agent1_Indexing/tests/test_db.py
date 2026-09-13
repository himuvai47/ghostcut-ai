import tempfile
import unittest
from pathlib import Path

from indexing_agent.db import IndexDB
from indexing_agent.models import MediaInfo, TimeRange, VisionAnalysis


class DatabaseTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            db = IndexDB(Path(td) / "test.db")
            media = MediaInfo(duration=20.0, width=1920, height=1080, fps=30.0, codec="h264")
            sid = db.upsert_source(
                source_uid="src_test", relative_path="a.mp4", absolute_path="C:/a.mp4",
                file_size=123, mtime_ns=1, fingerprint="sampled:x", fingerprint_mode="sampled",
                index_signature="sig", media=media, status="PROBED"
            )
            shot_id = db.add_shot(sid, "shot_test", 1, TimeRange(0, 10), "test")
            clip_id = db.add_clip(sid, shot_id, "clip_test", 1, TimeRange(0, 5))
            db.save_analysis(clip_id, VisionAnalysis(description="A watch.", analysis_confidence=.8), "model", "p1", 1)
            counts = db.source_counts(sid)
            self.assertEqual(counts["indexed"], 1)
            self.assertEqual(counts["failed"], 0)
            db.close()
