import logging
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.planner import _infer_match_mode, _assign_context_anchors, plan_segments
from video_creation_agent.usage import UsageLedger


def clip(cid, product='TAG Heuer', score=.7, source='s1', start=0.0, duration=6.0, desc=None):
    c=IndexedClip(
        clip_id=cid,source_id=source,source_video=f'{source}.mp4',source_path=f'{source}.mp4',
        start_time=float(start),end_time=float(start+duration),duration=float(duration),
        description=desc or f'{product} luxury watch product shot',primary_subject='watch',primary_product=product,
        secondary_products=[],objects=['watch'],people=[],actions=[],environment='studio',shot_type='close_up',
        camera_motion='static',content_type='product_closeup',visual_attributes=[],visual_concepts=['watch'],
        visible_text=[product],observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,
        representative_paths=[],representative_times=[],analysis_hash='h'+cid,face_status='no_face',
        face_scan_version=2,face_scan_mode='representative_frames')
    return c, Candidate(c,score,.1,.5,.9,score,face_status='no_face')


def req(mode='brand_filler', anchor='TAG Heuer', duration=6.0, entities=None, intent='conceptual', text='The brand creates the feeling of arrival'):
    return VisualRequirement(
        segment_id='vs_0001',start_word_id='w1',end_word_id='w2',narration_text=text,
        visual_intent=intent,requested_visual='abstract narration visual',primary_subject='TAG Heuer',
        required_entities=list(entities or []),required_attributes=['status symbolism'],preferred_shot_types=['close_up'],
        preferred_content_types=['product_closeup'],visual_concepts=['status'],avoid=['human face visible'],specificity='normal',
        match_mode=mode,context_anchor=anchor,timeline_start=0.0,timeline_end=duration)


class V115EditorialFreedomTests(unittest.TestCase):
    def pipeline(self):
        p=object.__new__(Agent2Pipeline)
        p.cfg=Agent2Config()
        p.log=logging.getLogger('v115')
        return p

    def test_abstract_brand_commentary_is_not_forced_literal(self):
        mode=_infer_match_mode('literal','conceptual','TAG Heuer creates the feeling of arrival',['TAG Heuer'],'high')
        self.assertEqual(mode,'brand_filler')
        mode2=_infer_match_mode(None,'conceptual','The buyer later realizes the signal meant less than expected',[],'normal')
        self.assertEqual(mode2,'brand_filler')

    def test_exact_model_detail_can_remain_literal(self):
        mode=_infer_match_mode('literal','literal_product_detail','The Carrera Heuer-01 chronograph caliber is visible',['Carrera Heuer-01'],'high')
        self.assertEqual(mode,'literal')

    def test_context_anchor_carries_main_brand_without_model_hijack(self):
        xs=[
            req('thematic','',entities=['TAG Heuer'],text='TAG Heuer is sold everywhere'),
            req('brand_filler','',entities=[],text='It creates a feeling of arrival'),
            req('literal','',entities=['Carrera Heuer-01'],intent='literal_product_detail',text='Carrera Heuer-01 movement'),
            req('brand_filler','',entities=[],text='The buyer pays for the name'),
            req('thematic','',entities=['Nomos'],text='Nomos offers manufacture credentials'),
        ]
        out=_assign_context_anchors(xs)
        self.assertEqual(out[0].context_anchor,'TAG Heuer')
        self.assertEqual(out[1].context_anchor,'TAG Heuer')
        self.assertEqual(out[2].context_anchor,'Carrera Heuer-01')
        self.assertEqual(out[3].context_anchor,'TAG Heuer')
        self.assertEqual(out[4].context_anchor,'Nomos')

    def test_brand_filler_requirement_drops_literal_sentence_constraints(self):
        p=self.pipeline()
        r=req('brand_filler','TAG Heuer',entities=['TAG Heuer'])
        effective=p._editorial_requirement(r)
        self.assertEqual(effective.match_mode,'brand_filler')
        self.assertEqual(effective.required_entities,[])
        self.assertEqual(effective.required_attributes,[])
        self.assertEqual(effective.preferred_shot_types,[])
        self.assertIn('TAG Heuer',effective.requested_visual)
        self.assertIn('intentional editorial filler',effective.requested_visual)

    def test_brand_filler_skips_qwen_and_prefers_same_brand_diverse_footage(self):
        p=self.pipeline(); ledger=UsageLedger(2,2); logs=[]
        generic,gcand=clip('generic','Omega',score=.92,source='s2',desc='generic luxury watch close up')
        tag,tcand=clip('tag','TAG Heuer',score=.72,source='s3',desc='TAG Heuer Carrera product shot')
        r=p._editorial_requirement(req('brand_filler','TAG Heuer',duration=6.0))
        with patch('video_creation_agent.pipeline.rank',return_value=[gcand,tcand]), \
             patch('video_creation_agent.pipeline.rerank') as rr:
            chosen,dec=p._pick_for_segment(r,3,[generic,tag],{'generic':[0,1],'tag':[1,0]},ledger,logs,req_emb=[1.0])
        rr.assert_not_called()
        self.assertTrue(chosen)
        self.assertEqual(chosen[0]['clip'].clip_id,'tag')
        self.assertTrue(chosen[0]['editorial_filler'])
        self.assertEqual(dec['match_mode'],'brand_filler')
        self.assertEqual(logs[-1]['fallback_mode'],'intentional_brand_filler')

    @patch('video_creation_agent.planner.chat')
    def test_planner_persists_model_match_mode_and_context_anchor(self,mock_chat):
        words=[]
        text=['TAG','Heuer','creates','the','feeling','of','arrival','today']
        for i,t in enumerate(text,1):
            words.append({'id':f'w{i:05d}','text':t,'start':float(i-1),'end':float(i)})
        mock_chat.return_value={'segments':[{'start_word_id':'w00001','end_word_id':'w00008','visual_intent':'conceptual',
            'requested_visual':'TAG Heuer watches','primary_subject':'TAG Heuer','required_entities':['TAG Heuer'],
            'required_attributes':[],'preferred_shot_types':[],'preferred_content_types':['product_closeup'],
            'visual_concepts':['luxury watch'],'avoid':[],'specificity':'normal','match_mode':'brand_filler'}]}
        out=plan_segments(Agent2Config(), ' '.join(text), words, 8.0)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0].match_mode,'brand_filler')
        self.assertEqual(out[0].context_anchor,'TAG Heuer')


if __name__=='__main__':
    unittest.main()
