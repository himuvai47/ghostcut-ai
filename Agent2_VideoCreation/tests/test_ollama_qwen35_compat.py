import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from video_creation_agent.ollama import chat

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class CompatHandler(BaseHTTPRequestHandler):
    calls = []
    mode = "empty_then_valid"

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n))
        type(self).calls.append(body)
        i = len(type(self).calls)

        if self.mode == "empty_then_valid":
            if i == 1:
                out = {
                    "message": {"content": "", "thinking": "I am still reasoning..."},
                    "done_reason": "stop",
                    "eval_count": 5000,
                }
            else:
                out = {"message": {"content": json.dumps({"answer": "ok"})}, "done_reason": "stop"}
        elif self.mode == "manual_json_fallback":
            if i == 1:
                out = {"message": {"content": "", "thinking": "reasoning only"}, "done_reason": "stop"}
            elif i == 2:
                out = {"message": {"content": "not json despite format"}, "done_reason": "stop"}
            else:
                out = {"message": {"content": "```json\n{\"answer\":\"fallback\"}\n```"}, "done_reason": "stop"}
        else:
            out = {"message": {"content": json.dumps({"wrong": "field"})}, "done_reason": "stop"}

        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class Qwen35CompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), CompatHandler)
        cls.port = cls.srv.server_port
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        CompatHandler.calls = []

    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def test_empty_thinking_turn_falls_back_to_no_think(self):
        CompatHandler.mode = "empty_then_valid"
        out = chat(self.base(), "qwen3.5:9b", [{"role": "user", "content": "x"}], SCHEMA, 64, 0)
        self.assertEqual(out, {"answer": "ok"})
        self.assertEqual(CompatHandler.calls[0]["think"], "low")
        self.assertIs(CompatHandler.calls[1]["think"], False)
        self.assertIn("format", CompatHandler.calls[1])

    def test_schema_format_regression_uses_prompt_json_fallback(self):
        CompatHandler.mode = "manual_json_fallback"
        out = chat(self.base(), "qwen3.5:9b", [{"role": "user", "content": "x"}], SCHEMA, 64, 0)
        self.assertEqual(out, {"answer": "fallback"})
        self.assertNotIn("format", CompatHandler.calls[2])
        self.assertIs(CompatHandler.calls[2]["think"], False)
        self.assertIn("OUTPUT CONTRACT", CompatHandler.calls[2]["messages"][-1]["content"])

    def test_schema_validation_rejects_wrong_shape(self):
        CompatHandler.mode = "wrong_schema"
        with self.assertRaises(RuntimeError) as cm:
            chat(self.base(), "qwen3.5:9b", [{"role": "user", "content": "x"}], SCHEMA, 64, 0)
        self.assertIn("missing required field", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
