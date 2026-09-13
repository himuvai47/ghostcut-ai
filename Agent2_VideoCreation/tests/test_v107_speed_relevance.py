import json
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.global_review import review
from video_creation_agent.models import Candidate, CandidateJudgment, IndexedClip, VisualRequirement
from video_creation_agent.ollama import chat_fast
from video_creation_agent.pipeline import Agent2Pipeline, review_risks
from video_creation_agent.rerank import rerank


def clip(cid='c1', product='watch', duration=6.0):
    return IndexedClip(
        clip_id=cid, source_id='s', source_video='v.mp4', source_path='v.mp4', start_time=0,end_time=duration,duration=duration,
        description=f'{product} product footage',primary_subject=product,primary_product=product,secondary_products=[],objects=['watch'],people=[],actions=[],environment='studio',shot_type='close_up',camera_motion='static',content_type='product_closeup',visual_attributes=['silver case'],visual_concepts=['watch'],visible_text=[],observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,representative_paths=[],representative_times=[],analysis_hash='h'+cid,face_status='no_face',face_scan_version=2,face_scan_mode='representative_frames')


def req(attrs=None):
    return VisualRequirement(segment_id='vs',start_word_id='w1',end_word_id='w2',narration_text='The dial color and hour markers catch your eye',visual_intent='literal_product_detail',requested_visual='watch dial',primary_subject='watch dial',required_entities=[],required_attributes=attrs or ['blue color','hour markers'],preferred_shot_types=['close_up'],preferred_content_types=['product_closeup'],visual_concepts=['dial'],avoid=['human face visible'],specificity='normal',timeline_start=0,timeline_end=3)


class V107Tests(unittest.TestCase):
    @patch('video_creation_agent.rerank.chat_fast')
    def test_fine_attributes_are_soft_in_reranker_prompt(self, mock_chat):
        mock_chat.return_value={'judgments':[{'clip_id':'c1','subject_match':'strong','attribute_match':'none','hard_requirement_failed':False,'decision':'acceptable_match','reason':'Related watch dial footage.'}]}
        out=rerank(Agent2Config(), req(), [Candidate(clip(),.8,.2,.6,.9,.8,face_status='no_face')])
        self.assertEqual(out[0].decision,'acceptable_match')
        prompt=mock_chat.call_args.args[2][0]['content']
        self.assertIn('GENERAL VISUAL RELEVANCE',prompt)
        self.assertIn('preferences ONLY',prompt)
        self.assertIn('soft_attributes',prompt)

    def test_global_review_risk_gate_ignores_normal_acceptable_edits(self):
        normal=[{'segment_id':'vs','clips':[{'match_level':'acceptable_match','use_number':1}]}]
        risky=[{'segment_id':'vs','clips':[{'match_level':'weak_match','use_number':1},{'match_level':'acceptable_match','use_number':2}]}]
        self.assertEqual(review_risks(normal),[])
        self.assertNotIn('vs:weak_match',review_risks(risky))
        self.assertIn('vs:repeat',review_risks(risky))

    @patch('video_creation_agent.global_review.chat_fast')
    def test_global_review_explicitly_ignores_fine_detail_misses(self, mock_chat):
        mock_chat.return_value={'issues':[]}
        review(Agent2Config(),[{'segment_id':'vs','clips':[]}])
        prompt=mock_chat.call_args.args[2][0]['content']
        self.assertIn('General relevance is enough',prompt)
        self.assertIn('hour markers',prompt)
        self.assertIn('When in doubt, KEEP',prompt)

    def test_judge_pool_uses_two_candidate_fast_batch(self):
        p=object.__new__(Agent2Pipeline); p.cfg=Agent2Config();
        pool=[Candidate(clip(str(i)),.8,.1,.5,.9,.8,face_status='no_face') for i in range(6)]
        def fake(_cfg,_req,batch):
            return [CandidateJudgment(c.clip.clip_id,'strong','none','partial','partial',False,[],'acceptable_match','Related.') for c in batch]
        with patch('video_creation_agent.pipeline.rerank',side_effect=fake) as rr:
            out=p._judge_pool(req(),pool,target_duration=3,allow_weak=True)
        self.assertEqual(rr.call_count,1)
        self.assertEqual(len(rr.call_args.args[2]),2)
        self.assertEqual(len(out),2)

    @patch('video_creation_agent.ollama._post')
    def test_chat_fast_never_enables_thinking(self,mock_post):
        mock_post.return_value={'message':{'content':json.dumps({'answer':'ok'})}}
        schema={'type':'object','properties':{'answer':{'type':'string'}},'required':['answer'],'additionalProperties':False}
        out=chat_fast('http://x','qwen3.5:9b',[{'role':'user','content':'x'}],schema,128,0)
        self.assertEqual(out,{'answer':'ok'})
        payload=mock_post.call_args.args[1]
        self.assertIs(payload['think'],False)
        self.assertEqual(payload['options']['num_predict'],128)

if __name__=='__main__': unittest.main()
