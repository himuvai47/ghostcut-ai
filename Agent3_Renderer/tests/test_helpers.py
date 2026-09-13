import tempfile, unittest
from pathlib import Path
from renderer_agent.config import Agent3Config
from renderer_agent.encoder import video_encode_args, choose_encoder
from renderer_agent.utils import ffconcat_quote
from renderer_agent.cache import clip_cache_key

class HelperTests(unittest.TestCase):
    def test_x264_args(self):
        a=video_encode_args(Agent3Config(),'libx264'); self.assertIn('libx264',a); self.assertIn('-crf',a)
    def test_nvenc_args(self):
        a=video_encode_args(Agent3Config(),'h264_nvenc'); self.assertIn('h264_nvenc',a); self.assertIn('-cq',a)
    def test_explicit_x264(self): self.assertEqual(choose_encoder(Agent3Config(encoder='libx264')),'libx264')
    def test_bad_encoder(self):
        with self.assertRaises(ValueError): choose_encoder(Agent3Config(encoder='wat'))
    def test_ffconcat_quote(self):
        q=ffconcat_quote(Path('/tmp/a b.mp4')); self.assertTrue(q.startswith("'")); self.assertIn('a b.mp4',q)
    def test_cache_changes_with_timing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.mp4'; p.write_bytes(b'abc')
            base={'source':str(p),'source_start':0,'source_end':3,'frame_count':90}
            k1=clip_cache_key(base,Agent3Config(),'libx264'); base['source_end']=4; k2=clip_cache_key(base,Agent3Config(),'libx264'); self.assertNotEqual(k1,k2)
    def test_cache_changes_with_source_mtime(self):
        import os,time
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.mp4'; p.write_bytes(b'abc')
            item={'source':str(p),'source_start':0,'source_end':3,'frame_count':90}
            k1=clip_cache_key(item,Agent3Config(),'libx264'); p.write_bytes(b'abcd'); k2=clip_cache_key(item,Agent3Config(),'libx264'); self.assertNotEqual(k1,k2)

if __name__=='__main__': unittest.main()
