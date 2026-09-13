import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from indexing_agent.config import IndexerConfig
from indexing_agent.db import IndexDB
from indexing_agent.face import FaceScanError, FaceScanResult
from indexing_agent.models import MediaInfo, TimeRange, VisionAnalysis
from indexing_agent.pipeline import IndexingPipeline


class _Logger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k): pass
    def exception(self, *a, **k): pass


class FaceBackfillTests(unittest.TestCase):
    def _workspace_with_clip(self, root: Path):
        ws = root / "ws"
        pipe = IndexingPipeline(IndexerConfig(), ws, _Logger())
        src = root / "video.mp4"
        src.write_bytes(b"dummy")
        media = MediaInfo(duration=10, width=100, height=100, fps=30, codec="h264")
        sid = pipe.db.upsert_source(source_uid="src_test", relative_path="video.mp4", absolute_path=str(src), file_size=5,
                                    mtime_ns=1, fingerprint="x", fingerprint_mode="sampled", index_signature="sig",
                                    media=media, status="INDEXED")
        shot = pipe.db.add_shot(sid, "shot", 1, TimeRange(0, 10), "test")
        clip = pipe.db.add_clip(sid, shot, "clip", 1, TimeRange(0, 5))
        frame = ws / "frames" / "src_test" / "clip" / "frame.jpg"
        frame.parent.mkdir(parents=True)
        Image.new("RGB", (32,32), (100,100,100)).save(frame)
        pipe.db.save_frame_evidence(clip, [2.5], [frame])
        pipe.db.save_analysis(clip, VisionAnalysis(description="A watch in hand"), "qwen2.5vl:7b", "vision-v3", 3)
        return pipe, clip

    def test_backfill_adds_face_metadata_and_exports_it(self):
        with tempfile.TemporaryDirectory() as td:
            pipe, clip = self._workspace_with_clip(Path(td))
            with patch("indexing_agent.pipeline.scan_face_status", return_value=FaceScanResult("no_face", [], "Hands only")):
                summary = pipe.backfill_faces()
            self.assertEqual(summary["clips_scanned"], 1)
            self.assertEqual(summary["face_status"]["no_face"], 1)
            row = pipe.db.conn.execute("SELECT face_status FROM clips WHERE id=?", (clip,)).fetchone()
            self.assertEqual(row["face_status"], "no_face")
            record = json.loads((pipe.exports_dir / "clips.jsonl").read_text().splitlines()[0])
            self.assertEqual(record["face_status"], "no_face")
            pipe.close()

    def test_scanner_failure_degrades_to_uncertain(self):
        with tempfile.TemporaryDirectory() as td:
            pipe, clip = self._workspace_with_clip(Path(td))
            with patch("indexing_agent.pipeline.scan_face_status", side_effect=FaceScanError("model unavailable")):
                summary = pipe.backfill_faces()
            self.assertEqual(summary["face_status"]["uncertain"], 1)
            row = pipe.db.conn.execute("SELECT status,face_status,face_scan_mode FROM clips WHERE id=?", (clip,)).fetchone()
            self.assertEqual(row["status"], "INDEXED")
            self.assertEqual(row["face_status"], "uncertain")
            self.assertEqual(row["face_scan_mode"], "representative_frames_scan_error")
            pipe.close()

    def test_second_backfill_skips_current_face_version(self):
        with tempfile.TemporaryDirectory() as td:
            pipe, _ = self._workspace_with_clip(Path(td))
            with patch("indexing_agent.pipeline.scan_face_status", return_value=FaceScanResult("no_face", [], "none")) as scan:
                first = pipe.backfill_faces()
                second = pipe.backfill_faces()
            self.assertEqual(first["clips_scanned"], 1)
            self.assertEqual(second["clips_scanned"], 0)
            self.assertEqual(scan.call_count, 1)
            pipe.close()
