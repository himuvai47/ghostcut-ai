import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.planner import plan_segments, _repair_planner_ranges


def words(n=12):
    return [
        {'id':f'w{i:05d}','text':f'word{i}','start':float(i-1),'end':float(i)}
        for i in range(1,n+1)
    ]


def seg(a,b,label='watch',mode='thematic'):
    return {
        'start_word_id':a,'end_word_id':b,
        'visual_intent':'literal_product','requested_visual':label,
        'primary_subject':label,'required_entities':[], 'required_attributes':[],
        'preferred_shot_types':[], 'preferred_content_types':['product_closeup'],
        'visual_concepts':['watch'],'avoid':[], 'specificity':'normal',
        'match_mode':mode,
    }


class PlannerRangeRepairTests(unittest.TestCase):
    def test_contiguous_ranges_are_unchanged(self):
        ws=words(8); ids=[w['id'] for w in ws]; pos={x:i for i,x in enumerate(ids)}
        xs=[seg('w00001','w00004','a'),seg('w00005','w00008','b')]
        out=_repair_planner_ranges(xs,ids,pos)
        self.assertEqual([(x['start_word_id'],x['end_word_id']) for x in out],
                         [('w00001','w00004'),('w00005','w00008')])

    @patch('video_creation_agent.planner.chat')
    def test_repairs_gap_and_preserves_segment_payload(self,mock_chat):
        ws=words(8)
        mock_chat.return_value={'segments':[
            seg('w00001','w00003','TAG dial','thematic'),
            seg('w00005','w00008','TAG bracelet','brand_filler'),
        ]}
        out=plan_segments(Agent2Config(),' '.join(w['text'] for w in ws),ws,8.0)
        self.assertEqual(out[0].start_word_id,'w00001')
        self.assertEqual(out[-1].end_word_id,'w00008')
        # Exact boundary may assign the orphan word to either adjacent segment, but
        # coverage must be contiguous with no missing/duplicate aligned word.
        pos={w['id']:i for i,w in enumerate(ws)}
        for a,b in zip(out,out[1:]):
            self.assertEqual(pos[b.start_word_id],pos[a.end_word_id]+1)
        self.assertEqual(out[-1].match_mode,'brand_filler')

    @patch('video_creation_agent.planner.chat')
    def test_repairs_overlap_and_leading_trailing_boundary_errors(self,mock_chat):
        ws=words(12)
        mock_chat.return_value={'segments':[
            seg('w00002','w00005','first'),
            seg('w00004','w00008','second'),
            seg('w00009','w00011','third'),
        ]}
        out=plan_segments(Agent2Config(),' '.join(w['text'] for w in ws),ws,12.0)
        pos={w['id']:i for i,w in enumerate(ws)}
        self.assertEqual(out[0].start_word_id,'w00001')
        self.assertEqual(out[-1].end_word_id,'w00012')
        for a,b in zip(out,out[1:]):
            self.assertEqual(pos[b.start_word_id],pos[a.end_word_id]+1)
        self.assertTrue(all(pos[x.end_word_id]>=pos[x.start_word_id] for x in out))

    def test_range_repair_never_creates_empty_segment(self):
        ws=words(6); ids=[w['id'] for w in ws]; pos={x:i for i,x in enumerate(ids)}
        xs=[seg('w00001','w00006','a'),seg('w00002','w00002','b'),seg('w00003','w00003','c')]
        out=_repair_planner_ranges(xs,ids,pos)
        bounds=[(pos[x['start_word_id']],pos[x['end_word_id']]) for x in out]
        self.assertEqual(bounds[0][0],0)
        self.assertEqual(bounds[-1][1],5)
        self.assertTrue(all(a<=b for a,b in bounds))
        for (_,b),(c,_) in zip(bounds,bounds[1:]):
            self.assertEqual(c,b+1)

if __name__=='__main__':
    unittest.main()
