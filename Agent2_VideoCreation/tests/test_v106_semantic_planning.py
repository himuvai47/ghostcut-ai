import logging
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.global_review import review
from video_creation_agent.models import IndexedClip, CandidateJudgment, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.usage import UsageLedger
from video_creation_agent.planner import plan_segments


def clip(cid, product):
    return IndexedClip(
        clip_id=cid, source_id='s', source_video=f'{product}.mp4', source_path=f'{product}.mp4',
        start_time=0, end_time=6, duration=6, description=f'{product} watch product shot',
        primary_subject=product, primary_product=product, secondary_products=[], objects=['watch'], people=[], actions=[],
        environment='studio', shot_type='close_up', camera_motion='static', content_type='product_closeup',
        visual_attributes=[], visual_concepts=['watch'], visible_text=[product], observed_details=[], uncertain_inferences=[],
        usable=True, analysis_confidence=.9, representative_paths=[], representative_times=[], analysis_hash='h'+cid,
        face_status='no_face', face_scan_version=2, face_scan_mode='representative_frames')


class DummyLedger:
    pass


class V106Tests(unittest.TestCase):
    @patch('video_creation_agent.planner.chat')
    def test_requested_visual_strips_framing_but_keeps_subject(self, mock_chat):
        mock_chat.return_value={'segments':[{
            'start_word_id':'w00001','end_word_id':'w00006','visual_intent':'process',
            'requested_visual':'Macro shot of a mechanical watch movement with visible gears and components moving',
            'primary_subject':'mechanical watch movement','required_entities':[],
            'required_attributes':['macro framing','visible gears'],'preferred_shot_types':['macro'],
            'preferred_content_types':['product_closeup'],'visual_concepts':['mechanical movement'],
            'avoid':[],'specificity':'normal'}]}
        words=[{'id':f'w{i:05d}','text':t,'start':(i-1)*.4,'end':i*.4} for i,t in enumerate(['Mechanical','watches','use','intricate','internal','components'],1)]
        xs=plan_segments(Agent2Config(),'Mechanical watches use intricate internal components.',words,2.4)
        self.assertNotIn('macro',xs[0].requested_visual.lower())
        self.assertIn('mechanical watch movement',xs[0].requested_visual.lower())
        self.assertNotIn('macro framing',[x.lower() for x in xs[0].required_attributes])
        self.assertIn('visible gears',[x.lower() for x in xs[0].required_attributes])
        self.assertEqual(xs[0].preferred_shot_types,['macro'])

    def test_collective_segment_can_use_one_clip_per_named_entity(self):
        p=object.__new__(Agent2Pipeline); p.cfg=Agent2Config(); p.log=logging.getLogger('v106')
        req=VisualRequirement(
            segment_id='vs_0005',start_word_id='w1',end_word_id='w10',narration_text='Watches range from rugged G-Shock designs to Apple Watch smartwatches.',
            visual_intent='comparison',requested_visual='G-Shock followed by Apple Watch',primary_subject='watches',
            required_entities=['G-Shock','Apple Watch'],required_attributes=[],preferred_shot_types=['close_up'],
            preferred_content_types=['product_closeup'],visual_concepts=['watch comparison'],avoid=['human face visible'],
            specificity='high',timeline_start=0,timeline_end=8)
        g=clip('g','G-Shock'); a=clip('a','Apple Watch')
        calls=[]
        def fake_pick(sub,*args,**kwargs):
            calls.append(sub.required_entities[0])
            c=g if sub.required_entities[0]=='G-Shock' else a
            j=CandidateJudgment(c.clip_id,'strong','strong','partial','strong',False,[],'strong_match','Correct named product.')
            return [{'clip':c,'judgment':j,'source_start':0.0,'source_end':4.0,'take':4.0,'confidence':.9,'use_number':1,'face_status':'no_face'}], {'segment_id':sub.segment_id,'status':'matched'}
        with patch.object(Agent2Pipeline,'_pick_for_segment',side_effect=fake_pick):
            chosen,dec=p._pick_collective_segment(req,0,[g,a],{},DummyLedger(),[])
        self.assertEqual(calls,['G-Shock','Apple Watch'])
        self.assertEqual([x['coverage_entity'] for x in chosen],['G-Shock','Apple Watch'])
        self.assertEqual(dec['missing_entities'],[])
        self.assertEqual(len(dec['selected']),2)

    def test_real_collective_pick_uses_distinct_relevant_clips(self):
        p=object.__new__(Agent2Pipeline); p.cfg=Agent2Config(retrieval_top_k=10,min_unused_retrieval_score=0.0); p.log=logging.getLogger('v106-real')
        req=VisualRequirement(
            segment_id='vs_0005',start_word_id='w1',end_word_id='w10',narration_text='Watches range from rugged G-Shock designs to Apple Watch smartwatches.',
            visual_intent='comparison',requested_visual='G-Shock and Apple Watch',primary_subject='watches',
            required_entities=['G-Shock','Apple Watch'],required_attributes=[],preferred_shot_types=[],preferred_content_types=['product_closeup'],
            visual_concepts=['watch comparison'],avoid=['human face visible'],specificity='high',timeline_start=0,timeline_end=8)
        g=clip('g','G-Shock'); a=clip('a','Apple Watch'); clips=[g,a]
        clip_embs={'g':[1.0,0.0],'a':[0.0,1.0]}
        def fake_embed(_url,_model,texts):
            out=[]
            for text in texts:
                out.append([1.0,0.0] if 'G-Shock' in text else [0.0,1.0])
            return out
        def fake_rerank(_cfg,sub,cands):
            target=sub.required_entities[0]
            ans=[]
            for c in cands:
                good=target.casefold() in c.clip.primary_product.casefold()
                ans.append(CandidateJudgment(c.clip.clip_id,'strong' if good else 'none','strong' if good else 'none','partial','strong' if good else 'none',not good,[],'strong_match' if good else 'reject','Correct named product.' if good else 'Wrong named product.'))
            return ans
        ledger=UsageLedger(2,2); retrieval=[]
        with patch('video_creation_agent.pipeline.embed',side_effect=fake_embed), patch('video_creation_agent.pipeline.rerank',side_effect=fake_rerank):
            chosen,dec=p._pick_for_segment(req,4,clips,clip_embs,ledger,retrieval)
        self.assertEqual([x['clip'].clip_id for x in chosen],['g','a'])
        self.assertEqual([x['coverage_entity'] for x in chosen],['G-Shock','Apple Watch'])
        self.assertEqual(dec['missing_entities'],[])
        self.assertEqual(ledger.info('g')['count'],1); self.assertEqual(ledger.info('a')['count'],1)

    @patch('video_creation_agent.global_review.chat_fast')
    def test_global_review_is_told_to_judge_multi_clip_coverage_collectively(self,mock_chat):
        mock_chat.return_value={'issues':[]}
        summary=[{'segment_id':'vs_0005','required_entities':['G-Shock','Apple Watch'],'clips':[{'coverage_entity':'G-Shock'},{'coverage_entity':'Apple Watch'}]}]
        review(Agent2Config(),summary)
        prompt=mock_chat.call_args.args[2][0]['content']
        self.assertIn('COLLECTIVELY',prompt)
        self.assertIn('separate sequential clips',prompt)


if __name__=='__main__': unittest.main()
