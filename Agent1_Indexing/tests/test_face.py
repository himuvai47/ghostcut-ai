import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from indexing_agent.config import IndexerConfig
from indexing_agent.face import FACE_SCAN_VERSION, FaceScanError, FaceScanResult, scan_face_status


class FaceScanTests(unittest.TestCase):
    def _image(self, root: Path, name: str) -> Path:
        p = root / name
        Image.new("RGB", (32, 32), (120, 120, 120)).save(p)
        return p

    def test_no_face_accepts_hands_body_without_face(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            p = self._image(root, "f.jpg")
            env = {"message": {"content": json.dumps({
                "face_status": "no_face",
                "evidence_frame_indices": [],
                "reason": "Hands and torso only; no facial features visible."
            })}}
            with patch("indexing_agent.face._json_request", return_value=env):
                result = scan_face_status([p], IndexerConfig())
            self.assertEqual(result.face_status, "no_face")
            self.assertEqual(result.evidence_frame_indices, [])

    def test_face_visible_preserves_evidence_timestamp_mapping(self):
        result = FaceScanResult("face_visible", [2], "Profile face visible")
        payload = result.evidence_payload([1.0, 2.5, 4.0])
        self.assertEqual(payload["frames"], [{"frame_index": 2, "timestamp": 2.5}])

    def test_invalid_status_retries_then_fails(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._image(Path(td), "f.jpg")
            env = {"message": {"content": json.dumps({
                "face_status": "person",
                "evidence_frame_indices": [],
                "reason": "bad enum"
            })}}
            with patch("indexing_agent.face._json_request", return_value=env):
                with self.assertRaises(FaceScanError):
                    scan_face_status([p], IndexerConfig(model_retries=1))


    def test_face_request_uses_portable_json_mode(self):
        with tempfile.TemporaryDirectory() as td:
            p = self._image(Path(td), "f.jpg")
            env = {"message": {"content": json.dumps({
                "face_status": "no_face",
                "evidence_frame_indices": [],
                "reason": "No face visible."
            })}}
            with patch("indexing_agent.face._json_request", return_value=env) as req:
                scan_face_status([p], IndexerConfig())
            payload = req.call_args.args[1]
            self.assertEqual(payload["format"], "json")
            self.assertEqual(payload["options"]["num_ctx"], IndexerConfig().ollama_num_ctx)

    def test_face_scan_version_is_two(self):
        self.assertEqual(FACE_SCAN_VERSION, 2)
