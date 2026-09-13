import unittest
from renderer_agent.config import Agent3Config
from renderer_agent.render_plan import build_render_plan

def timeline(clips,duration=10):
    return {'audio':{'source':'a.wav','duration':duration},'segments':[{'segment_id':'s','status':'matched','clips':clips}]}

class PlanTests(unittest.TestCase):
    def setUp(self): self.cfg=Agent3Config(width=640,height=360,fps=30,encoder='libx264')
    def test_single_clip_frames(self):
        t=timeline([{'clip_id':'c','source':'v','source_start':1,'source_end':4,'timeline_start':0,'timeline_end':3}],3)
        p=build_render_plan(t,self.cfg,'libx264'); self.assertEqual(p['target_frames'],90); self.assertEqual(p['items'][0]['frame_count'],90)
    def test_middle_gap_inserted(self):
        t=timeline([{'clip_id':'a','source':'v','source_start':0,'source_end':3,'timeline_start':0,'timeline_end':3},{'clip_id':'b','source':'v','source_start':3,'source_end':6,'timeline_start':4,'timeline_end':7}],7)
        p=build_render_plan(t,self.cfg,'libx264'); self.assertEqual([i['type'] for i in p['items']],['clip','gap','clip']); self.assertEqual(p['items'][1]['frame_count'],30)
    def test_end_gap_inserted(self):
        t=timeline([{'clip_id':'a','source':'v','source_start':0,'source_end':3,'timeline_start':0,'timeline_end':3}],5)
        p=build_render_plan(t,self.cfg,'libx264'); self.assertEqual(p['items'][-1]['type'],'gap'); self.assertEqual(p['items'][-1]['frame_count'],60)
    def test_rounding_uses_global_frames(self):
        t=timeline([{'clip_id':'a','source':'v','source_start':0,'source_end':1.001,'timeline_start':0,'timeline_end':1.001}],1.001)
        p=build_render_plan(t,self.cfg,'libx264'); self.assertEqual(sum(i['frame_count'] for i in p['items']),p['target_frames'])
    def test_output_profile(self):
        p=build_render_plan(timeline([{'clip_id':'a','source':'v','source_start':0,'source_end':3,'timeline_start':0,'timeline_end':3}],3),self.cfg,'libx264')
        self.assertEqual(p['output']['width'],640); self.assertEqual(p['output']['fps'],30)

if __name__=='__main__': unittest.main()
