import json, tempfile, unittest
from pathlib import Path
from renderer_agent.config import Agent3Config

class ConfigTests(unittest.TestCase):
    def test_defaults(self):
        c=Agent3Config(); self.assertEqual((c.width,c.height,c.fps),(1920,1080,30)); self.assertEqual(c.encoder,'auto')
    def test_load_override(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json'; p.write_text(json.dumps({'width':1280,'height':720,'fps':24}))
            c=Agent3Config.load(p); self.assertEqual((c.width,c.height,c.fps),(1280,720,24))
    def test_unknown_key_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json'; p.write_text(json.dumps({'bogus':1}))
            with self.assertRaises(ValueError): Agent3Config.load(p)
    def test_invalid_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json'; p.write_text(json.dumps({'width':0}))
            with self.assertRaises(ValueError): Agent3Config.load(p)
    def test_signature_stable(self):
        self.assertEqual(Agent3Config().signature(),Agent3Config().signature())

if __name__=='__main__': unittest.main()
