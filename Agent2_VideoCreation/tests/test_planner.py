import unittest
from unittest.mock import patch
from video_creation_agent.config import Agent2Config
from video_creation_agent.planner import plan_segments

class PlannerTests(unittest.TestCase):
    @patch('video_creation_agent.planner.chat')
    def test_meaningful_segments_cover_words(self,mock_chat):
        mock_chat.return_value={'segments':[
            {'start_word_id':'w00001','end_word_id':'w00003','visual_intent':'literal_product','requested_visual':'Rolex watch hero shot','primary_subject':'Rolex watch','required_entities':['Rolex'],'required_attributes':[],'preferred_shot_types':['close_up'],'preferred_content_types':['product_closeup'],'visual_concepts':['watch'],'avoid':[],'specificity':'high'},
            {'start_word_id':'w00004','end_word_id':'w00006','visual_intent':'literal_product_detail','requested_visual':'close-up of the dial','primary_subject':'watch dial','required_entities':['Rolex'],'required_attributes':['dial clearly visible'],'preferred_shot_types':['close_up'],'preferred_content_types':['product_closeup'],'visual_concepts':['dial design'],'avoid':[],'specificity':'high'}]}
        words=[{'id':f'w{i:05d}','text':t,'start':(i-1)*.5,'end':i*.5} for i,t in enumerate(['Rolex','makes','watches','The','dial','matters'],1)]
        xs=plan_segments(Agent2Config(),'Rolex makes watches. The dial matters.',words,3.2)
        self.assertEqual(len(xs),1); self.assertEqual(xs[0].timeline_start,0.0); self.assertEqual(xs[-1].timeline_end,3.2); self.assertGreaterEqual(xs[0].duration,3.0)
        self.assertIn('human face visible',xs[0].avoid)


    @patch('video_creation_agent.planner.chat')
    def test_removes_invented_production_techniques(self,mock_chat):
        mock_chat.return_value={'segments':[{'start_word_id':'w00001','end_word_id':'w00006','visual_intent':'process','requested_visual':'Transparent X-ray style animation or cutaway view of a mechanical watch movement, showing gears and springs in motion.','primary_subject':'mechanical watch movement','required_entities':[],'required_attributes':['x-ray animation','gears visible'],'preferred_shot_types':['macro'],'preferred_content_types':['product_closeup'],'visual_concepts':['mechanical movement'],'avoid':[],'specificity':'normal'}]}
        words=[{'id':f'w{i:05d}','text':t,'start':(i-1)*.4,'end':i*.4} for i,t in enumerate(['Mechanical','watches','use','intricate','internal','components'],1)]
        xs=plan_segments(Agent2Config(),'Mechanical watches use intricate internal components.',words,2.4)
        self.assertNotIn('x-ray',xs[0].requested_visual.lower())
        self.assertNotIn('animation',xs[0].requested_visual.lower())
        self.assertNotIn('x-ray animation',[a.lower() for a in xs[0].required_attributes])
        self.assertIn('gears visible',[a.lower() for a in xs[0].required_attributes])

    @patch('video_creation_agent.planner.chat')
    def test_repairs_leading_gap(self,mock_chat):
        mock_chat.return_value={'segments':[{'start_word_id':'w00002','end_word_id':'w00002','visual_intent':'literal_product','requested_visual':'watch','primary_subject':'watch','required_entities':[],'required_attributes':[],'preferred_shot_types':[],'preferred_content_types':[],'visual_concepts':[],'avoid':[],'specificity':'normal'}]}
        words=[{'id':'w00001','text':'a','start':0,'end':.2},{'id':'w00002','text':'b','start':.2,'end':.4}]
        xs=plan_segments(Agent2Config(),'a b',words,.5)
        self.assertEqual(xs[0].start_word_id,'w00001')
        self.assertEqual(xs[0].end_word_id,'w00002')

if __name__=='__main__': unittest.main()
