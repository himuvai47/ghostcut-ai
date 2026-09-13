import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, CandidateJudgment, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.qc import validate_timeline


def mkclip(cid,duration,score=.8):
    c=IndexedClip(cid,'s','v.mp4','v.mp4',0,duration,duration,'watch footage','watch','watch',[],['watch'],[],[],'studio','close_up','static','product_closeup',[],['watch'],[],[],[],True,.9,[],[],'h'+cid,face_status='no_face',face_scan_version=2,face_scan_mode='representative_frames')
    return Candidate(c,score,.2,.6,.9,score,face_status='no_face')


def req(duration):
    return VisualRequirement('vs','w1','w2','watch narration','literal_product','watch','watch',[],[],[],['product_closeup'],['watch'],['human face visible'],'normal',timeline_start=0,timeline_end=duration)


def judgment(c,level='acceptable_match'):
    return CandidateJudgment(c.clip.clip_id,'strong','partial','partial','strong',False,[],level,'Relevant watch footage.')


class V108PacingTests(unittest.TestCase):
    def pipeline(self):
        p=object.__new__(Agent2Pipeline)
        p.cfg=Agent2Config()
        return p

    def test_planned_take_never_exceeds_eight_and_avoids_tiny_remainder(self):
        p=self.pipeline()
        self.assertAlmostEqual(p._planned_take(7.0,12.0),7.0)
        self.assertAlmostEqual(p._planned_take(9.5,12.0),6.5)  # leaves a legal 3s second cut
        self.assertAlmostEqual(p._planned_take(20.0,12.0),8.0)

    def test_sub_three_source_clip_is_not_a_legal_cut(self):
        p=self.pipeline()
        self.assertEqual(p._planned_take(6.0,2.9),0.0)

    def test_longer_clip_preferred_inside_same_match_level(self):
        p=self.pipeline()
        short,long=mkclip('short',3.2,.9),mkclip('long',7.5,.8)
        out=p._candidate_order([judgment(short),judgment(long)],[short,long])
        self.assertEqual(out[0][0].clip.clip_id,'long')

    def test_collective_requires_room_for_three_seconds_per_entity(self):
        p=self.pipeline()
        r=req(5.5); r.required_entities=['G-Shock','Apple Watch']
        self.assertEqual(p._collective_entities(r),[])
        r.timeline_end=6.2
        self.assertEqual(p._collective_entities(r),['G-Shock','Apple Watch'])

    def test_qc_rejects_flash_cut_and_overlong_cut(self):
        c=mkclip('c',12).clip
        flash={'audio':{'duration':10},'segments':[{'segment_id':'s','timeline_start':0,'timeline_end':10,'status':'matched','clips':[{'clip_id':'c','source_start':0,'source_end':2,'timeline_start':0,'timeline_end':2,'face_status':'no_face'}]}]}
        q=validate_timeline(flash,{'c':c},2,3,8)
        self.assertFalse(q['passed']); self.assertTrue(any('min 3.00s' in x for x in q['issues']))
        long={'audio':{'duration':10},'segments':[{'segment_id':'s','timeline_start':0,'timeline_end':10,'status':'matched','clips':[{'clip_id':'c','source_start':0,'source_end':9,'timeline_start':0,'timeline_end':9,'face_status':'no_face'}]}]}
        q=validate_timeline(long,{'c':c},2,3,8)
        self.assertFalse(q['passed']); self.assertTrue(any('max 8.00s' in x for x in q['issues']))

    def test_judge_pool_prefers_one_longer_clip_and_stops_fast(self):
        p=self.pipeline()
        pool=[mkclip('a',7.0,.78),mkclip('b',3.2,.88),mkclip('c',6.0,.77)]
        def fake(_cfg,_req,batch):
            return [judgment(x) for x in batch]
        with patch('video_creation_agent.pipeline.rerank',side_effect=fake) as rr:
            out=p._judge_pool(req(7.0),pool,target_duration=7.0,allow_weak=True)
        self.assertEqual(rr.call_count,1)
        self.assertEqual(out[0][0].clip.clip_id,'a')

if __name__=='__main__': unittest.main()
