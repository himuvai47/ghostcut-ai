import unittest
from video_creation_agent.qc import validate_timeline
from video_creation_agent.models import IndexedClip

def clip(cid='c'):
 return IndexedClip(cid,'s','v','v',0,10,10,'d','watch','watch',[],[],[],[],'','close_up','static','product_closeup',[],[],[],[],[],True,.9,[],[],'h')

class QCTests(unittest.TestCase):
 def test_uncertain_face_status_is_allowed(self):
  c=clip(); tl={'audio':{'duration':5},'segments':[{'segment_id':'s1','timeline_start':0,'timeline_end':5,'status':'matched','clips':[{'clip_id':'c','source_start':0,'source_end':5,'face_status':'uncertain'}]}]}
  q=validate_timeline(tl,{'c':c}); self.assertTrue(q['passed']); self.assertEqual(q['uncertain_face_clips_used'],1)
 def test_visible_face_fails(self):
  c=clip(); tl={'audio':{'duration':5},'segments':[{'segment_id':'s1','timeline_start':0,'timeline_end':5,'status':'matched','clips':[{'clip_id':'c','source_start':0,'source_end':5,'face_status':'face_visible'}]}]}
  q=validate_timeline(tl,{'c':c}); self.assertFalse(q['passed']); self.assertTrue(any('face gate' in x for x in q['issues']))
 def test_more_than_two_uses_fails(self):
  c=clip(); segs=[]
  for i in range(3): segs.append({'segment_id':f's{i}','timeline_start':i,'timeline_end':i+1,'status':'matched','clips':[{'clip_id':'c','source_start':0,'source_end':1,'face_status':'no_face'}]})
  q=validate_timeline({'audio':{'duration':3},'segments':segs},{'c':c},2); self.assertFalse(q['passed']); self.assertTrue(any('used 3 times' in x for x in q['issues']))

if __name__=='__main__': unittest.main()
