import tempfile
import unittest
from pathlib import Path

from indexing_agent.db import IndexDB
from indexing_agent.models import MediaInfo, TimeRange, VisionAnalysis
from indexing_agent.qc import qc_source


class QCTests(unittest.TestCase):
    def test_non_editorial_unusable_passes(self):
        with tempfile.TemporaryDirectory() as td:
            db = IndexDB(Path(td) / "qc.db")
            media = MediaInfo(duration=5.0, width=1920, height=1080, fps=30.0, codec="h264")
            sid = db.upsert_source(
                source_uid="src", relative_path="a.mp4", absolute_path="a.mp4", file_size=1,
                mtime_ns=1, fingerprint="x", fingerprint_mode="sampled", index_signature="sig",
                media=media, status="PROBED",
            )
            shot = db.add_shot(sid, "shot", 1, TimeRange(0, 5), "test")
            clip = db.add_clip(sid, shot, "clip", 1, TimeRange(0, 5))
            db.save_frame_evidence(clip, [2.5], [Path("frame.jpg")])
            db.save_analysis(
                clip,
                VisionAnalysis(
                    description="Thanks for watching screen.", primary_subject="end card",
                    shot_type="text_screen", content_type="end_screen", usable=False,
                    exclusion_reason="end_screen", confidence_level="high", analysis_confidence=.90,
                    confidence_basis=["non-editorial screen classification is visually explicit"],
                    objects=["end card"],
                ),
                "model", "vision-v3", 3,
            )
            report = qc_source(db, sid)
            self.assertTrue(report["passed"])
            self.assertEqual(report["unusable_clip_count"], 1)
            self.assertEqual(report["confidence_score_summary"]["min"], 0.9)
            db.close()
