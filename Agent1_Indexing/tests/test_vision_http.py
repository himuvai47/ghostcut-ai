import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

from indexing_agent.config import IndexerConfig
from indexing_agent.vision import analyze_clip


PAYLOAD = {
    "description": "Close-up of a watch dial with clearly visible hour markers.",
    "primary_subject": "watch dial",
    "secondary_subjects": [],
    "primary_product": "wristwatch with dark dial",
    "secondary_products": [],
    "objects": ["watch", "dial"],
    "people": [],
    "actions": [],
    "environment": "product setup",
    "shot_type": "close_up",
    "camera_motion": "unknown",
    "content_type": "product_detail",
    "visual_attributes": ["dark dial"],
    "visual_concepts": ["product design"],
    "visible_text": [],
    "observed_details": ["The dial fills much of the frame."],
    "uncertain_inferences": [],
    "brand": "unknown",
    "model_name": "unknown",
    "subject_focus": "dial",
    "shot_angle": "front",
    "editorial_role": "product_beauty",
    "visual_energy": "calm",
    "lighting_style": "neutral",
    "composition": "single_subject",
    "quality_flags": [],
    "usable": True,
    "exclusion_reason": "none",
}


class Handler(BaseHTTPRequestHandler):
    received = None

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        Handler.received = json.loads(self.rfile.read(length))
        body = json.dumps({"message": {"role": "assistant", "content": json.dumps(PAYLOAD)}, "done": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class VisionHTTPTests(unittest.TestCase):
    def test_multiframe_structured_request_and_software_confidence(self):
        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for i in range(2):
                    p = Path(td) / f"f{i}.jpg"
                    Image.new("RGB", (32, 32), (i * 20, 10, 10)).save(p)
                    paths.append(p)
                cfg = IndexerConfig(ollama_url=f"http://127.0.0.1:{server.server_port}", model_retries=1)
                result = analyze_clip(paths, cfg)
                self.assertEqual(result.description, PAYLOAD["description"])
                self.assertEqual(result.primary_product, "wristwatch with dark dial")
                self.assertTrue(result.confidence_basis)
                req = Handler.received
                self.assertEqual(req["model"], "qwen2.5vl:7b")
                self.assertEqual(len(req["messages"][1]["images"]), 2)
                self.assertIsInstance(req["format"], dict)
                self.assertFalse(req["stream"])
                self.assertIn("content_type", req["format"]["properties"])
                self.assertIn("editorial_role", req["format"]["properties"])
                self.assertIn("visual_energy", req["format"]["properties"])
                self.assertNotIn("confidence_level", req["format"]["properties"])
                self.assertNotIn("visual_quality_score", req["format"]["properties"])
                self.assertIn("Do NOT output a confidence", req["messages"][0]["content"])
        finally:
            server.shutdown()
            server.server_close()


class RetryHandler(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        RetryHandler.calls.append(payload)
        n = len(RetryHandler.calls)
        if n < 3:
            # Simulate the exact real-world failure: Ollama envelope is valid, assistant JSON is truncated.
            assistant = '{"description":"A long response that never closes'
            response = {"message": {"role": "assistant", "content": assistant}, "done": True, "done_reason": "length"}
        else:
            response = {"message": {"role": "assistant", "content": json.dumps(PAYLOAD)}, "done": True, "done_reason": "stop"}
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class VisionRetryTests(unittest.TestCase):
    def test_truncated_json_retries_compactly_and_reduces_images(self):
        RetryHandler.calls = []
        server = HTTPServer(("127.0.0.1", 0), RetryHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as td:
                paths = []
                for i in range(5):
                    p = Path(td) / f"f{i}.jpg"
                    Image.new("RGB", (32, 32), (i * 20, 10, 10)).save(p)
                    paths.append(p)
                cfg = IndexerConfig(
                    ollama_url=f"http://127.0.0.1:{server.server_port}",
                    model_retries=3,
                    retry_image_cap=3,
                    final_retry_image_cap=2,
                    ollama_num_predict=3072,
                )
                result = analyze_clip(paths, cfg)
                self.assertEqual(result.primary_product, "wristwatch with dark dial")
                self.assertEqual(len(RetryHandler.calls), 3)
                self.assertEqual([len(c["messages"][1]["images"]) for c in RetryHandler.calls], [5, 3, 2])
                self.assertEqual(RetryHandler.calls[0]["options"]["num_predict"], 3072)
                self.assertIn("maxLength", RetryHandler.calls[0]["format"]["properties"]["description"])
                self.assertIn("maxItems", RetryHandler.calls[0]["format"]["properties"]["observed_details"])
                self.assertIn("FINAL COMPACT RETRY", RetryHandler.calls[2]["messages"][1]["content"])
        finally:
            server.shutdown()
            server.server_close()
