import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import logging

from PIL import Image

from indexing_agent.config import IndexerConfig
from indexing_agent.db import IndexDB
from indexing_agent.models import MediaInfo, TimeRange, VisionAnalysis, validate_vision_payload
from indexing_agent.pipeline import IndexingPipeline


class EditorialVisionTests(unittest.TestCase):
    def payload(self):
        return {
            "description": "Macro view of a TAG Heuer Carrera dial and chronograph details.",
            "primary_subject": "TAG Heuer Carrera wristwatch",
            "secondary_subjects": [],
            "primary_product": "TAG Heuer Carrera wristwatch",
            "secondary_products": [],
            "objects": ["watch", "dial", "chronograph subdials"],
            "people": [],
            "actions": [],
            "environment": "dark studio product setup",
            "shot_type": "macro",
            "camera_motion": "orbit",
            "content_type": "product_closeup",
            "visual_attributes": ["metal bracelet", "dark dial"],
            "visual_concepts": ["watch design"],
            "visible_text": ["TAG HEUER", "CARRERA"],
            "observed_details": ["The dial fills most of the frame."],
            "uncertain_inferences": [],
            "brand": "TAG Heuer",
            "model_name": "Carrera",
            "subject_focus": "dial",
            "shot_angle": "three_quarter",
            "editorial_role": "product_beauty",
            "visual_energy": "calm",
            "lighting_style": "dark",
            "composition": "single_subject",
            "quality_flags": [],
            "usable": True,
            "exclusion_reason": "none",
        }

    def test_editorial_profile_is_derived(self):
        result = validate_vision_payload(self.payload())
        # Dial close-ups normalize to technical detail for consistent filler sequencing.
        self.assertEqual(result.editorial_role, "technical_detail")
        self.assertEqual(result.brand, "TAG Heuer")
        self.assertEqual(result.model_name, "Carrera")
        self.assertIn("carrera", result.visual_family)
        self.assertIn("dial", result.visual_family)
        self.assertGreaterEqual(result.visual_quality_score, 0.75)
        self.assertGreaterEqual(result.editorial_usefulness_score, 0.75)
        self.assertEqual(result.editorial_profile_version, 1)

    def test_visual_family_changes_with_focus(self):
        a = validate_vision_payload(self.payload())
        p = self.payload()
        p["subject_focus"] = "movement"
        p["editorial_role"] = "movement_detail"
        p["description"] = "Macro view of the TAG Heuer Carrera mechanical movement."
        b = validate_vision_payload(p)
        self.assertNotEqual(a.visual_family, b.visual_family)
        self.assertIn("movement", b.visual_family)

    def test_quality_defects_reduce_scores(self):
        clean = validate_vision_payload(self.payload())
        p = self.payload()
        p["quality_flags"] = ["out_of_focus", "motion_blur"]
        degraded = validate_vision_payload(p)
        self.assertLess(degraded.visual_quality_score, clean.visual_quality_score)
        self.assertLess(degraded.editorial_usefulness_score, clean.editorial_usefulness_score)

    def test_default_v121_signature_is_recognized(self):
        cfg = IndexerConfig()
        self.assertEqual(
            cfg.legacy_v121_index_signature(),
            "26102f465548a38200aa63c0760a4b84266b7a898450a81fe760ac7a694694cb",
        )
        self.assertNotEqual(cfg.index_signature(), cfg.legacy_v121_index_signature())


class EditorialBackfillDBTests(unittest.TestCase):

    def test_pipeline_editorial_backfill_reuses_cached_frames_and_preserves_face(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logger = logging.getLogger("test-editorial-backfill")
            pipe = IndexingPipeline(IndexerConfig(), root, logger)
            try:
                source_path = root / "a.mp4"
                media = MediaInfo(duration=5, width=1920, height=1080, fps=30, codec="h264")
                source_id = pipe.db.upsert_source(
                    source_uid="src", relative_path="a.mp4", absolute_path=str(source_path),
                    file_size=1, mtime_ns=1, fingerprint="fp", fingerprint_mode="sampled",
                    index_signature=pipe.legacy_v121_signature, media=media, status="INDEXED",
                )
                shot_id = pipe.db.add_shot(source_id, "shot", 1, TimeRange(0, 5), "test")
                clip_id = pipe.db.add_clip(source_id, shot_id, "clip", 1, TimeRange(0, 5))
                frame = root / "frame.jpg"
                Image.new("RGB", (32, 32), (20, 20, 20)).save(frame)
                pipe.db.save_frame_evidence(clip_id, [2.5], [frame])
                pipe.db.save_analysis(clip_id, VisionAnalysis(description="Old clip"), "qwen2.5vl:7b", "vision-v3", 3)
                pipe.db.save_face_scan(clip_id, status="no_face", version=2, mode="test", evidence={"frames": []})
                upgraded = VisionAnalysis(
                    description="TAG Heuer Carrera dial beauty shot.", brand="TAG Heuer", model_name="Carrera",
                    subject_focus="dial", shot_angle="front", editorial_role="technical_detail",
                    visual_family="carrera__technical_detail__dial", visual_quality_score=0.91,
                    editorial_usefulness_score=0.94,
                )
                with patch("indexing_agent.pipeline.analyze_clip", return_value=upgraded) as mocked:
                    summary = pipe.backfill_editorial()
                self.assertEqual(summary["clips_upgraded"], 1)
                self.assertEqual(summary["clips_failed_preserved"], 0)
                self.assertEqual(summary["representative_frames_regenerated"], 0)
                mocked.assert_called_once()
                row = pipe.db.conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
                self.assertEqual(row["face_status"], "no_face")
                self.assertEqual(row["analysis_version"], 4)
                src = pipe.db.conn.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
                self.assertEqual(src["index_signature"], pipe.index_signature)
            finally:
                pipe.close()

    def test_v3_clip_is_selected_for_editorial_backfill_without_losing_face_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            db = IndexDB(Path(td) / "idx.db")
            try:
                media = MediaInfo(duration=10, width=1920, height=1080, fps=30, codec="h264")
                source_id = db.upsert_source(
                    source_uid="src", relative_path="a.mp4", absolute_path=str(Path(td) / "a.mp4"),
                    file_size=1, mtime_ns=1, fingerprint="fp", fingerprint_mode="sampled",
                    index_signature="legacy", media=media, status="INDEXED",
                )
                shot_id = db.add_shot(source_id, "shot", 1, TimeRange(0, 10), "test")
                clip_id = db.add_clip(source_id, shot_id, "clip", 1, TimeRange(0, 5))
                old = VisionAnalysis(description="Old indexed watch clip.")
                db.save_analysis(clip_id, old, "qwen2.5vl:7b", "vision-v3", 3)
                db.save_face_scan(
                    clip_id, status="no_face", version=2, mode="test", evidence={"frames": []}
                )
                rows = db.indexed_clips_needing_editorial(4, "vision-v4-editorial-index")
                self.assertEqual(len(rows), 1)
                new = VisionAnalysis(
                    description="Upgraded watch clip.", brand="TAG Heuer", model_name="Carrera",
                    subject_focus="dial", shot_angle="front", editorial_role="technical_detail",
                    visual_family="carrera__technical_detail__dial", visual_quality_score=0.9,
                    editorial_usefulness_score=0.92,
                )
                db.save_analysis(clip_id, new, "qwen2.5vl:7b", "vision-v4-editorial-index", 4)
                row = db.conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
                self.assertEqual(row["face_status"], "no_face")
                self.assertEqual(row["face_scan_version"], 2)
                self.assertEqual(len(db.indexed_clips_needing_editorial(4, "vision-v4-editorial-index")), 0)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
