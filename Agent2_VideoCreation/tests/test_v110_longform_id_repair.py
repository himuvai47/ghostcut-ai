import unittest
from unittest.mock import patch
from video_creation_agent.config import Agent2Config
from video_creation_agent.planner import plan_segments


def seg(a,b,label='watch'):
    return {
        'start_word_id':a,'end_word_id':b,
        'visual_intent':'literal_product','requested_visual':label,
        'primary_subject':label,'required_entities':[], 'required_attributes':[],
        'preferred_shot_types':[], 'preferred_content_types':['product_closeup'],
        'visual_concepts':['watch'],'avoid':[], 'specificity':'normal'
    }


def words(n=8):
    return [{'id':f'w{i:05d}','text':f'word{i}','start':float(i-1),'end':float(i)} for i in range(1,n+1)]


class LongFormIdRepairTests(unittest.TestCase):
    @patch('video_creation_agent.planner.chat')
    def test_repairs_numeric_word_id_format_variants(self,mock_chat):
        mock_chat.return_value={'segments':[seg('word_1','word_4','dial'),seg('w5','w8','bracelet')]}
        xs=plan_segments(Agent2Config(),' '.join(x['text'] for x in words()),words(),8.0)
        self.assertEqual([(x.start_word_id,x.end_word_id) for x in xs],[('w00001','w00004'),('w00005','w00008')])

    @patch('video_creation_agent.planner.chat')
    def test_repairs_unresolvable_start_to_expected_contiguous_word(self,mock_chat):
        mock_chat.return_value={'segments':[seg('w00001','w00004','dial'),seg('not-a-real-id','w00008','bracelet')]}
        xs=plan_segments(Agent2Config(),' '.join(x['text'] for x in words()),words(),8.0)
        self.assertEqual(xs[1].start_word_id,'w00005')
        self.assertEqual(xs[1].end_word_id,'w00008')

    @patch('video_creation_agent.planner.chat')
    def test_repairs_invalid_end_from_next_valid_start_and_final_word(self,mock_chat):
        mock_chat.return_value={'segments':[seg('w00001','bad-end','dial'),seg('w00005','also-bad','bracelet')]}
        xs=plan_segments(Agent2Config(),' '.join(x['text'] for x in words()),words(),8.0)
        self.assertEqual([(x.start_word_id,x.end_word_id) for x in xs],[('w00001','w00004'),('w00005','w00008')])

    @patch('video_creation_agent.planner.chat')
    def test_valid_gap_is_repaired_by_v117_range_stitching(self,mock_chat):
        mock_chat.return_value={'segments':[seg('w00002','w00004','dial'),seg('w00005','w00008','bracelet')]}
        xs=plan_segments(Agent2Config(),' '.join(x['text'] for x in words()),words(),8.0)
        self.assertEqual(xs[0].start_word_id,'w00001')
        self.assertEqual(xs[-1].end_word_id,'w00008')


if __name__=='__main__':
    unittest.main()
