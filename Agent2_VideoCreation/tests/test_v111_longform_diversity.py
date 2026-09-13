import logging
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, CandidateJudgment, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.planner import plan_segments
from video_creation_agent.qc import validate_timeline
from video_creation_agent.usage import UsageLedger


def seg(a,b,label='TAG Heuer watch'):
    return {
        'start_word_id':a,'end_word_id':b,
        'visual_intent':'literal_product','requested_visual':label,
        'primary_subject':label,'required_entities':['TAG Heuer'], 'required_attributes':[],
        'preferred_shot_types':[], 'preferred_content_types':['product_closeup'],
        'visual_concepts':['watch'],'avoid':[], 'specificity':'high'
    }


def words(n=260):
    return [{'id':f'w{i:05d}','text':f'word{i}','start':float(i-1),'end':float(i)} for i in range(1,n+1)]


def clip(cid, duration=6.0, product='TAG Heuer', score=.7):
    c=IndexedClip(
        clip_id=cid,source_id='s',source_video=f'{cid}.mp4',source_path=f'{cid}.mp4',start_time=0,end_time=duration,duration=duration,
        description=f'{product} watch B-roll',primary_subject='watch',primary_product=product,secondary_products=[],objects=['watch'],people=[],actions=[],environment='studio',shot_type='close_up',camera_motion='static',content_type='product_closeup',visual_attributes=[],visual_concepts=['watch'],visible_text=[product],observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,representative_paths=[],representative_times=[],analysis_hash='h'+cid,face_status='no_face',face_scan_version=2,face_scan_mode='representative_frames')
    return c, Candidate(c,score,.1,.5,.9,score,face_status='no_face')


def req(duration=6.0):
    return VisualRequirement('vs','w1','w2','Narration about a watch brand and collector opinion','literal_product','TAG Heuer watch','watch',['Nomos watch'],[],[],['product_closeup'],['watch'],['human face visible'],'high',timeline_start=0,timeline_end=duration)


class V111LongFormDiversityTests(unittest.TestCase):
    @patch('video_creation_agent.planner.chat')
    def test_four_minute_plan_is_deterministically_split_into_many_visual_beats(self,mock_chat):
        ws=words(260)
        # Simulate the exact problem: Qwen returns only eight very large semantic ideas.
        bounds=[]
        start=1
        for i in range(8):
            end=260 if i==7 else start+31
            bounds.append(seg(f'w{start:05d}',f'w{end:05d}'))
            start=end+1
        mock_chat.return_value={'segments':bounds}
        xs=plan_segments(Agent2Config(),' '.join(w['text'] for w in ws),ws,260.0)
        self.assertGreaterEqual(len(xs),24)
        self.assertLessEqual(max(x.duration for x in xs),10.05)
        self.assertGreaterEqual(min(x.duration for x in xs),2.95)
        self.assertEqual(xs[0].start_word_id,'w00001')
        self.assertEqual(xs[-1].end_word_id,'w00260')
        # Word ranges remain exactly contiguous.
        pos={w['id']:i for i,w in enumerate(ws)}
        expected=0
        for x in xs:
            self.assertEqual(pos[x.start_word_id],expected)
            expected=pos[x.end_word_id]+1
        self.assertEqual(expected,260)

    def test_broad_fallback_can_use_related_watch_broll_when_exact_nomoss_is_absent(self):
        p=object.__new__(Agent2Pipeline); p.cfg=Agent2Config(); p.log=logging.getLogger('v111')
        ledger=UsageLedger(2,2); c,cand=clip('generic',6.0,'TAG Heuer',.62); logs=[]
        seen={}
        def fake_rerank(_cfg,relaxed,batch):
            seen['entities']=list(relaxed.required_entities)
            # Simulate overly literal reranker rejection; deterministic rescue must still use topical B-roll.
            return [CandidateJudgment(batch[0].clip.clip_id,'partial','none','partial','partial',False,[],'reject','Exact Nomos not shown.')]
        with patch('video_creation_agent.pipeline.embed',return_value=[[1.0]]), \
             patch('video_creation_agent.pipeline.rank',return_value=[cand]), \
             patch('video_creation_agent.pipeline.rerank',side_effect=fake_rerank):
            out=p._relaxed_gap_fill(req(),4,[c],{},ledger,logs,6.0,set(),False)
        self.assertEqual(seen['entities'],[])
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['clip'].clip_id,'generic')
        self.assertEqual(out[0]['judgment'].decision,'acceptable_match')
        self.assertTrue(out[0]['gap_filler'])

    def test_rich_library_blocks_repeats_while_many_unused_clips_exist(self):
        p=object.__new__(Agent2Pipeline); p.cfg=Agent2Config(min_unused_clips_before_repeat=12); p.log=logging.getLogger('v111-repeat')
        clips=[]
        for i in range(20): clips.append(clip(f'c{i}',6.0)[0])
        ledger=UsageLedger(2,2)
        ledger.mark('c0',0)
        self.assertGreaterEqual(p._unused_legal_count(clips,ledger),12)

    def test_qc_rejects_matched_segment_with_large_uncovered_span(self):
        c=clip('c',6.0)[0]
        tl={'audio':{'duration':8.0},'segments':[{
            'segment_id':'vs_1','timeline_start':0.0,'timeline_end':8.0,'status':'matched',
            'clips':[{'clip_id':'c','source_start':0.0,'source_end':4.0,'timeline_start':0.0,'timeline_end':4.0,'face_status':'no_face'}]
        }]}
        q=validate_timeline(tl,{'c':c},2,3,8)
        self.assertFalse(q['passed'])
        self.assertTrue(any('uncovered' in x for x in q['issues']))


if __name__=='__main__': unittest.main()
