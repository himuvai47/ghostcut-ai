import logging
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, CandidateJudgment, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.usage import UsageLedger


def mk_clip(cid, duration, score=.62):
    c=IndexedClip(
        clip_id=cid,source_id='s',source_video=f'{cid}.mp4',source_path=f'{cid}.mp4',
        start_time=0,end_time=duration,duration=duration,description='TAG Heuer watch B-roll',
        primary_subject='watch',primary_product='TAG Heuer',secondary_products=[],objects=['watch'],
        people=[],actions=[],environment='studio',shot_type='close_up',camera_motion='static',
        content_type='product_closeup',visual_attributes=[],visual_concepts=['watch'],visible_text=['TAG Heuer'],
        observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,
        representative_paths=[],representative_times=[],analysis_hash='h'+cid,
        face_status='no_face',face_scan_version=2,face_scan_mode='representative_frames')
    return c, Candidate(c,score,.1,.5,.9,score,face_status='no_face')


def requirement(duration=4.47):
    return VisualRequirement(
        segment_id='vs_0027',start_word_id='w1',end_word_id='w2',
        narration_text='Collectors moved toward other watch choices over time',visual_intent='conceptual',
        requested_visual='timeline or graphic showing community shift',primary_subject='watch market',
        required_entities=[],required_attributes=[],preferred_shot_types=[],preferred_content_types=[],
        visual_concepts=['watch collecting'],avoid=['human face visible'],specificity='normal',
        timeline_start=0,timeline_end=duration)


def rejected(cand):
    return CandidateJudgment(cand.clip.clip_id,'partial','none','partial','partial',False,[],
                             'reject','Too literal for requested graphic.')


class V113CoverageCompletionTests(unittest.TestCase):
    def pipeline(self):
        p=object.__new__(Agent2Pipeline)
        p.cfg=Agent2Config()
        p.log=logging.getLogger('v113')
        return p

    def test_447_gap_scans_past_two_short_candidates_without_extra_qwen(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); logs=[]
        c1,a=mk_clip('short1',3.20,.75)
        c2,b=mk_clip('short2',3.40,.73)
        c3,c=mk_clip('long_enough',5.50,.68)
        ranked=[a,b,c]
        with patch('video_creation_agent.pipeline.embed',return_value=[[1.0]]), \
             patch('video_creation_agent.pipeline.rank',return_value=ranked), \
             patch('video_creation_agent.pipeline.rerank',return_value=[rejected(a),rejected(b)]) as rr:
            out=p._relaxed_gap_fill(requirement(),27,[c1,c2,c3],{},ledger,logs,4.47,set(),False)
        self.assertEqual(rr.call_count,1)
        self.assertEqual(len(rr.call_args.args[2]),2)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['clip'].clip_id,'long_enough')
        self.assertAlmostEqual(out[0]['take'],4.47,places=2)
        self.assertTrue(out[0]['gap_filler'])
        self.assertEqual(ledger.info('long_enough')['count'],1)
        self.assertGreaterEqual(logs[-1]['deterministic_scan_count'],3)

    def test_deterministic_completion_can_use_two_legal_cuts(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); logs=[]
        # For a 7s remainder, 4s then 3s is legal and should fully cover it.
        cs=[]; ranked=[]
        for cid,dur,score in [('a',4.0,.7),('b',3.5,.68),('c',3.2,.66)]:
            c,ca=mk_clip(cid,dur,score); cs.append(c); ranked.append(ca)
        with patch('video_creation_agent.pipeline.embed',return_value=[[1.0]]), \
             patch('video_creation_agent.pipeline.rank',return_value=ranked), \
             patch('video_creation_agent.pipeline.rerank',return_value=[rejected(ranked[0]),rejected(ranked[1])]):
            out=p._relaxed_gap_fill(requirement(7.0),5,cs,{},ledger,logs,7.0,set(),False)
        self.assertAlmostEqual(sum(x['take'] for x in out),7.0,places=2)
        self.assertGreaterEqual(len(out),2)

if __name__=='__main__': unittest.main()
