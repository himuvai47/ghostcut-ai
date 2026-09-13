import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import VisualRequirement
from video_creation_agent.pipeline import review_risks
from video_creation_agent.planner import _derive_split_child, plan_segments


def planner_segment(a,b):
    return {
        'start_word_id':a,'end_word_id':b,
        'visual_intent':'comparison',
        'requested_visual':'split screen or side-by-side comparison of two watches',
        'primary_subject':'luxury watches',
        'required_entities':['TAG Heuer','Nomos','buyers','METAS certification'],
        'required_attributes':['certification text overlay'],
        'preferred_shot_types':[],
        'preferred_content_types':['product_closeup'],
        'visual_concepts':['luxury watches'],
        'avoid':[],
        'specificity':'high',
    }


class V112BeatSpecificSemanticsTests(unittest.TestCase):
    def test_split_child_uses_its_own_narration_and_drops_generic_hard_entities(self):
        parent=VisualRequirement(
            'vs','w00001','w00020','broad parent','comparison',
            'split screen or side-by-side comparison of two watches','luxury watches',
            ['TAG Heuer','Nomos','buyers','METAS certification'],['certification text overlay'],[],['product_closeup'],['watch'],[],'high')
        chunk=[
            {'id':'w00001','text':'TAG','start':0.0,'end':0.5},
            {'id':'w00002','text':'Heuer','start':0.5,'end':1.0},
            {'id':'w00003','text':'buyers','start':1.0,'end':1.5},
            {'id':'w00004','text':'recognize','start':1.5,'end':2.0},
            {'id':'w00005','text':'the','start':2.0,'end':2.5},
            {'id':'w00006','text':'name','start':2.5,'end':3.0},
        ]
        child=_derive_split_child(parent,chunk)
        self.assertIn('TAG Heuer',child.required_entities)
        self.assertNotIn('buyers',child.required_entities)
        self.assertNotIn('Nomos',child.required_entities)
        self.assertNotIn('METAS certification',child.required_entities)
        self.assertNotIn('split screen',child.requested_visual.casefold())
        self.assertIn('buyers recognize the name',child.requested_visual.casefold())

    @patch('video_creation_agent.planner.chat')
    def test_long_parent_becomes_distinct_narration_specific_queries(self,mock_chat):
        texts=(
            ['TAG','Heuer','is','sold','in','airport','stores','worldwide']+
            ['buyers','pay','for','the','name','and','marketing','visibility']+
            ['mechanical','movement','value','becomes','important','to','serious','collectors']+
            ['Nomos','offers','different','manufacture','credentials','at','lower','prices']
        )
        words=[]
        for i,t in enumerate(texts,1):
            words.append({'id':f'w{i:05d}','text':t,'start':float(i-1),'end':float(i)})
        mock_chat.return_value={'segments':[planner_segment('w00001',f'w{len(words):05d}')]} 
        xs=plan_segments(Agent2Config(target_visual_segment_seconds=8,max_visual_segment_seconds=10), ' '.join(texts), words, float(len(words)))
        self.assertGreaterEqual(len(xs),4)
        reqs=[x.requested_visual for x in xs]
        self.assertEqual(len(reqs),len(set(reqs)))
        self.assertTrue(all('split screen' not in x.casefold() for x in reqs))
        self.assertIn('TAG Heuer',xs[0].required_entities)
        self.assertTrue(any('Nomos' in x.required_entities for x in xs))
        self.assertTrue(all('buyers' not in x.required_entities for x in xs))

    def test_weak_related_match_does_not_trigger_global_review(self):
        weak=[{'segment_id':'vs_0001','clips':[{'match_level':'weak_match','use_number':1}]}]
        repeated=[{'segment_id':'vs_0002','clips':[{'match_level':'acceptable_match','use_number':2}]}]
        self.assertEqual(review_risks(weak),[])
        self.assertEqual(review_risks(repeated),['vs_0002:repeat'])

if __name__=='__main__':
    unittest.main()
