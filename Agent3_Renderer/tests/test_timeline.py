import tempfile, unittest, json
from pathlib import Path
from renderer_agent.timeline_reader import load_timeline, flatten_clips, validate_structure

def tl():
    return {'audio':{'source':'a.wav','duration':10},'segments':[
        {'segment_id':'s1','status':'matched','clips':[{'clip_id':'c1','source':'v.mp4','source_start':1,'source_end':4,'timeline_start':0,'timeline_end':3}]},
        {'segment_id':'s2','status':'no_match','clips':[]},
        {'segment_id':'s3','status':'matched','clips':[{'clip_id':'c2','source':'v.mp4','source_start':5,'source_end':8,'timeline_start':4,'timeline_end':7}]}]}

class TimelineTests(unittest.TestCase):
    def test_flatten_sorted(self):
        x=tl(); x['segments'].reverse(); clips=flatten_clips(x); self.assertEqual([c['clip_id'] for c in clips],['c1','c2'])
    def test_no_match_ignored(self): self.assertEqual(len(flatten_clips(tl())),2)
    def test_valid_structure(self): self.assertEqual(validate_structure(tl()),[])
    def test_overlap_detected(self):
        x=tl(); x['segments'][2]['clips'][0]['timeline_start']=2.5; self.assertTrue(any('overlaps' in i for i in validate_structure(x)))
    def test_bad_source_range(self):
        x=tl(); x['segments'][0]['clips'][0]['source_end']=.5; self.assertTrue(any('source_end' in i for i in validate_structure(x)))
    def test_load_requires_audio_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'t.json'; p.write_text(json.dumps({'audio':{'duration':1},'segments':[{}]}))
            with self.assertRaises(ValueError): load_timeline(p)
    def test_load_requires_segments(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'t.json'; p.write_text(json.dumps({'audio':{'source':'a','duration':1},'segments':[]}))
            with self.assertRaises(ValueError): load_timeline(p)

if __name__=='__main__': unittest.main()
