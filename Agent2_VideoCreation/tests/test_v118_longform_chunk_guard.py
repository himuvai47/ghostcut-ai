import logging
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import VisualRequirement
from video_creation_agent.planner import (
    _planner_range_anomaly, _chunk_word_ranges, _assign_context_anchors,
    _plan_raw_segments
)


def words(n=700):
    out=[]
    for i in range(1,n+1):
        text=f'word{i}' + ('.' if i%25==0 else '')
        out.append({'id':f'w{i:05d}','text':text,'start':(i-1)*0.4,'end':i*0.4})
    return out


def seg(a,b,entity='TAG Heuer',mode='thematic'):
    return {
        'start_word_id':a,'end_word_id':b,
        'visual_intent':'literal_product','requested_visual':f'{entity} watch footage',
        'primary_subject':entity,'required_entities':[entity] if entity else [],'required_attributes':[],
        'preferred_shot_types':[],'preferred_content_types':['product_closeup'],
        'visual_concepts':['watch'],'avoid':[],'specificity':'normal','match_mode':mode,
    }


def req(entity, mode='brand_filler', text='abstract narration'):
    return VisualRequirement(
        segment_id='vs',start_word_id='w1',end_word_id='w2',narration_text=text,
        visual_intent='conceptual',requested_visual='watch footage',primary_subject=entity,
        required_entities=[entity] if entity else [],required_attributes=[],preferred_shot_types=[],
        preferred_content_types=[],visual_concepts=[],avoid=['human face visible'],specificity='normal',
        match_mode=mode)


class LongformChunkGuardTests(unittest.TestCase):
    def test_detects_massive_missing_prefix_instead_of_stretching_first_segment(self):
        ws=words(700); ids=[w['id'] for w in ws]; pos={x:i for i,x in enumerate(ids)}
        xs=[seg('w00458','w00469'),seg('w00470','w00700')]
        reason=_planner_range_anomaly(xs,ids,pos)
        self.assertIn('leading words',reason)
        self.assertIn('457',reason)

    def test_longform_chunk_ranges_cover_every_word_once(self):
        ws=words(700); ranges=_chunk_word_ranges(ws)
        self.assertGreaterEqual(len(ranges),3)
        self.assertEqual(ranges[0][0],0)
        self.assertEqual(ranges[-1][1],699)
        for (_,b),(c,_) in zip(ranges,ranges[1:]):
            self.assertEqual(c,b+1)
        self.assertTrue(all((b-a+1)<=235 for a,b in ranges))

    @patch('video_creation_agent.planner.chat')
    def test_longform_plans_in_focused_chunks(self,mock_chat):
        ws=words(700)
        def reply(*args,**kwargs):
            prompt=args[2][0]['content']
            ids=[]
            import re
            ids=re.findall(r'\bw\d{5}\b',prompt.split('WORD IDS:')[-1])
            # _word_text may include one id once; preserve first/last occurrence.
            uniq=[]
            for x in ids:
                if x not in uniq: uniq.append(x)
            return {'segments':[seg(uniq[0],uniq[-1])]}
        mock_chat.side_effect=reply
        out=_plan_raw_segments(Agent2Config(),' '.join(w['text'] for w in ws),ws)
        self.assertGreater(mock_chat.call_count,1)
        self.assertEqual(out[0]['start_word_id'],'w00001')
        self.assertEqual(out[-1]['end_word_id'],'w00700')
        pos={w['id']:i for i,w in enumerate(ws)}
        for a,b in zip(out,out[1:]):
            self.assertEqual(pos[b['start_word_id']],pos[a['end_word_id']]+1)

    def test_generic_cta_phrases_do_not_hijack_context_anchor(self):
        xs=[
            req('TAG Heuer','thematic','TAG Heuer remains the topic'),
            req('Tell me','brand_filler','Tell me in the comments which watch you own'),
            req('luxury counter','thematic','before you decide at a luxury counter'),
            req('Swiss watch brands','thematic','the next brand is Swiss'),
        ]
        out=_assign_context_anchors(xs)
        self.assertEqual([x.context_anchor for x in out],['TAG Heuer']*4)

if __name__=='__main__':
    unittest.main()
