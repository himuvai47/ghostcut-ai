import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, CandidateJudgment, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.usage import UsageLedger


def clip(cid='c1', duration=5.0, score=.7):
    c=IndexedClip(
        clip_id=cid,source_id='s',source_video='v.mp4',source_path='v.mp4',start_time=0,end_time=duration,duration=duration,
        description='generally related watch B-roll',primary_subject='watch',primary_product='watch',secondary_products=[],objects=['watch'],people=[],actions=[],environment='studio',shot_type='close_up',camera_motion='static',content_type='product_closeup',visual_attributes=[],visual_concepts=['watch'],visible_text=[],observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,representative_paths=[],representative_times=[],analysis_hash='h'+cid,face_status='no_face',face_scan_version=2,face_scan_mode='representative_frames')
    return c, Candidate(c,score,.1,.5,.9,score,face_status='no_face')


def req(duration=6.51,specificity='normal'):
    return VisualRequirement(
        segment_id='vs_0004',start_word_id='w1',end_word_id='w2',narration_text='The bracelet and clasp matter when you look at construction and finishing',visual_intent='literal_product_detail',requested_visual='watch bracelet and clasp',primary_subject='watch bracelet and clasp',required_entities=['bracelet','clasp'],required_attributes=['construction','finishing'],preferred_shot_types=['close_up'],preferred_content_types=['product_closeup'],visual_concepts=['watch details'],avoid=['human face visible'],specificity=specificity,timeline_start=0,timeline_end=duration)


def judge(c,level='acceptable_match'):
    return CandidateJudgment(c.clip.clip_id,'strong','partial','partial','strong',False,[],level,'Generally related watch B-roll.')


class V109GapFillTests(unittest.TestCase):
    def pipeline(self):
        p=object.__new__(Agent2Pipeline)
        p.cfg=Agent2Config()
        return p

    def test_relaxed_gap_fill_uses_one_unused_related_clip_for_3s_gap(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); c,cand=clip('filler',5.0,.62); logs=[]
        with patch('video_creation_agent.pipeline.embed',return_value=[[1.0,0.0]]), \
             patch('video_creation_agent.pipeline.rank',return_value=[cand]), \
             patch('video_creation_agent.pipeline.rerank',return_value=[judge(cand)]) as rr:
            out=p._relaxed_gap_fill(req(),3,[c],{},ledger,logs,3.255,set(),False)
        self.assertEqual(len(out),1)
        self.assertAlmostEqual(out[0]['take'],3.255,places=3)
        self.assertTrue(out[0]['gap_filler'])
        self.assertEqual(rr.call_count,1)
        self.assertEqual(len(rr.call_args.args[2]),1)
        self.assertEqual(ledger.info('filler')['count'],1)
        self.assertEqual(logs[-1]['fallback_mode'],'general_gap_fill_unused')

    def test_gap_fill_judges_at_most_two_candidates(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); logs=[]
        cs=[]; cands=[]
        for i in range(4):
            c,ca=clip(f'c{i}',5.0,.7-i*.02); cs.append(c); cands.append(ca)
        def fake(_cfg,_req,batch): return [judge(x) for x in batch]
        with patch('video_creation_agent.pipeline.embed',return_value=[[1.0]]), \
             patch('video_creation_agent.pipeline.rank',return_value=cands), \
             patch('video_creation_agent.pipeline.rerank',side_effect=fake) as rr:
            out=p._relaxed_gap_fill(req(),1,cs,{},ledger,logs,6.0,set(),False)
        self.assertLessEqual(len(rr.call_args.args[2]),2)
        self.assertGreaterEqual(sum(x['take'] for x in out),3.0)

    def test_broad_fallback_relaxes_named_entity_after_exact_match_failed(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); c,cand=clip('brand',5.0,.7); logs=[]
        r=req(4.0,'high'); r.required_entities=['Rolex']; r.primary_subject='Rolex watch'
        seen={}
        def fake(_cfg,relaxed,batch):
            seen['entities']=list(relaxed.required_entities)
            return [judge(batch[0])]
        with patch('video_creation_agent.pipeline.embed',return_value=[[1.0]]), \
             patch('video_creation_agent.pipeline.rank',return_value=[cand]), \
             patch('video_creation_agent.pipeline.rerank',side_effect=fake):
            p._relaxed_gap_fill(r,1,[c],{},ledger,logs,4.0,set(),False)
        self.assertEqual(seen['entities'],[])

    def test_collective_segment_fills_missing_entity_gap_with_related_broll(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); c1,_=clip('bracelet',3.255); c2,_=clip('filler',4.0)
        chosen1={'clip':c1,'judgment':CandidateJudgment('bracelet','strong','partial','partial','strong',False,[],'acceptable_match','Bracelet shown.'),'source_start':0.0,'source_end':3.255,'take':3.255,'confidence':.75,'use_number':1,'face_status':'no_face'}
        filler={'clip':c2,'judgment':CandidateJudgment('filler','strong','partial','partial','strong',False,[],'acceptable_match','Related watch B-roll.'),'source_start':0.0,'source_end':3.255,'take':3.255,'confidence':.72,'use_number':1,'face_status':'no_face','gap_filler':True}
        calls=[]
        def fake_pick(sub,*args,**kwargs):
            calls.append(sub.required_entities[0])
            if sub.required_entities[0]=='bracelet': return [chosen1],{'status':'matched'}
            return [],{'status':'no_match'}
        with patch.object(p,'_pick_for_segment',side_effect=fake_pick), \
             patch.object(p,'_relaxed_gap_fill',side_effect=[[filler],[]]) as gf:
            selected,dec=p._pick_collective_segment(req(),2,[c1,c2],{},ledger,[])
        self.assertEqual(calls,['bracelet','clasp'])
        self.assertEqual(len(selected),2)
        self.assertAlmostEqual(dec['coverage_shortfall'],0.0,places=3)
        self.assertTrue(selected[1]['gap_filler'])
        self.assertEqual(gf.call_count,1)

if __name__=='__main__': unittest.main()
