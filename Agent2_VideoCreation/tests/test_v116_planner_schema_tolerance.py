import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.planner import (
    SCHEMA, _normalize_content_types, _normalize_shot_types, plan_segments,
)

class V116PlannerSchemaToleranceTests(unittest.TestCase):
    def test_soft_preference_schema_accepts_arbitrary_strings(self):
        item=SCHEMA['properties']['segments']['items']['properties']
        self.assertNotIn('enum', item['preferred_content_types']['items'])
        self.assertNotIn('enum', item['preferred_shot_types']['items'])

    def test_comparison_and_common_aliases_normalize_safely(self):
        self.assertEqual(_normalize_content_types(['comparison']), ['product_closeup','product_detail'])
        self.assertEqual(_normalize_content_types(['beauty shot','wrist shot','totally_new_label']), ['product_closeup','product_handling'])
        self.assertEqual(_normalize_shot_types(['close-up','medium wide','cinematic orbit']), ['close_up','medium_wide'])

    @patch('video_creation_agent.planner.chat')
    def test_plan_segments_survives_model_comparison_label_and_uses_longform_token_floor(self, mock_chat):
        words=[]
        toks=['TAG','Heuer','versus','Nomos','for','value','and','movement']
        for i,t in enumerate(toks,1):
            words.append({'id':f'w{i:05d}','text':t,'start':float(i-1),'end':float(i)})
        mock_chat.return_value={'segments':[{
            'start_word_id':'w00001','end_word_id':'w00008','visual_intent':'comparison',
            'requested_visual':'TAG Heuer and Nomos watches','primary_subject':'watch comparison',
            'required_entities':['TAG Heuer','Nomos'],'required_attributes':[],
            'preferred_shot_types':['close-up'],'preferred_content_types':['comparison'],
            'visual_concepts':['watch value comparison'],'avoid':[], 'specificity':'high','match_mode':'literal'
        }]}
        cfg=Agent2Config(qwen_num_predict=1024)
        out=plan_segments(cfg,' '.join(toks),words,8.0)
        self.assertEqual(out[0].preferred_content_types,['product_closeup','product_detail'])
        self.assertEqual(out[0].preferred_shot_types,['close_up'])
        self.assertEqual(mock_chat.call_args.args[4],4096)

if __name__=='__main__':
    unittest.main()
