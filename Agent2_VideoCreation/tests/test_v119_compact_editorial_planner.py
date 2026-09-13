import re
import unittest
from unittest.mock import patch

from video_creation_agent.config import Agent2Config
from video_creation_agent.planner import (
    _paced_word_ranges, _compact_mode, _compact_plan_batch, plan_segments,
    PLANNER_VERSION,
)


def words(n=700, step=.4):
    out=[]
    for i in range(1,n+1):
        text=f'word{i}' + ('.' if i%20==0 else '')
        out.append({'id':f'w{i:05d}','text':text,'start':(i-1)*step,'end':i*step})
    return out


class V119CompactPlannerTests(unittest.TestCase):
    def test_paced_ranges_cover_every_word_once(self):
        ws=words(700)
        rr=_paced_word_ranges(ws,8.0,10.0,3.0)
        self.assertGreater(len(rr),25)
        self.assertEqual(rr[0][0],0)
        self.assertEqual(rr[-1][1],699)
        for (_,b),(c,_) in zip(rr,rr[1:]):
            self.assertEqual(c,b+1)
        for a,b in rr[:-1]:
            dur=ws[b]['end']-ws[a]['start']
            self.assertLessEqual(dur,10.05)
            self.assertGreaterEqual(dur,2.95)

    def test_mode_aliases_are_tolerant(self):
        self.assertEqual(_compact_mode('filler'),'brand_filler')
        self.assertEqual(_compact_mode('topic'),'thematic')
        self.assertEqual(_compact_mode('exact'),'literal')

    @patch('video_creation_agent.planner.chat')
    def test_longform_uses_compact_batches_without_word_ids(self,mock_chat):
        ws=words(700)
        def reply(*args,**kwargs):
            prompt=args[2][0]['content']
            self.assertNotIn('WORD IDS:',prompt)
            ids=re.findall(r'^(b\d{4}) \|',prompt,re.M)
            return {'beats':[{'beat_id':x,'mode':'brand_filler','anchor':'TAG Heuer',
                              'visual':'TAG Heuer watch B-roll','entities':['TAG Heuer']} for x in ids]}
        mock_chat.side_effect=reply
        out=plan_segments(Agent2Config(),'ignored',ws,280.0)
        self.assertGreater(mock_chat.call_count,1)
        self.assertGreater(len(out),25)
        self.assertTrue(all(r.match_mode=='brand_filler' for r in out))
        self.assertTrue(all(r.context_anchor=='TAG Heuer' for r in out))
        pos={w['id']:i for i,w in enumerate(ws)}
        self.assertEqual(out[0].start_word_id,'w00001')
        self.assertEqual(out[-1].end_word_id,'w00700')
        for a,b in zip(out,out[1:]):
            self.assertEqual(pos[b.start_word_id],pos[a.end_word_id]+1)

    @patch('video_creation_agent.planner.chat')
    def test_bad_compact_batch_splits_then_single_beat_falls_back(self,mock_chat):
        mock_chat.side_effect=RuntimeError('malformed json')
        beats=[{'beat_id':f'b{i:04d}','narration':'abstract TAG Heuer commentary'} for i in range(1,4)]
        out=_compact_plan_batch(Agent2Config(),beats)
        self.assertEqual(len(out),3)
        self.assertTrue(all(x['mode']=='brand_filler' for x in out))
        self.assertGreaterEqual(mock_chat.call_count,5)  # batch -> split -> singles

    @patch('video_creation_agent.planner.chat')
    def test_missing_or_junk_anchor_does_not_break_full_plan(self,mock_chat):
        ws=words(400)
        def reply(*args,**kwargs):
            prompt=args[2][0]['content']
            ids=re.findall(r'^(b\d{4}) \|',prompt,re.M)
            rows=[]
            for j,x in enumerate(ids):
                rows.append({'beat_id':x,'mode':'filler','anchor':'Tell me' if j else 'TAG Heuer',
                             'visual':'watch footage','entities':['TAG Heuer'] if j==0 else []})
            return {'beats':rows}
        mock_chat.side_effect=reply
        out=plan_segments(Agent2Config(),'ignored',ws,160.0)
        self.assertTrue(out)
        self.assertTrue(all(r.context_anchor=='TAG Heuer' for r in out))

    def test_planner_version_changed(self):
        self.assertEqual(PLANNER_VERSION,'planner-v13-compact-editorial-planner')

if __name__=='__main__':
    unittest.main()
