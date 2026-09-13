import json, sqlite3, tempfile, unittest
from pathlib import Path
from video_creation_agent.models import IndexedClip, VisualRequirement
from video_creation_agent.retrieval import cosine, lexical, structured, clip_text
from video_creation_agent.usage import UsageLedger
from video_creation_agent.source_window import choose_window
from video_creation_agent.audio_alignment import extract_asr_words, align_script


def clip(cid='c1',**kw):
    d=dict(clip_id=cid,source_id='s1',source_video='a.mp4',source_path='C:/a.mp4',start_time=10,end_time=20,duration=10,
    description='close up blue Rolex watch dial',primary_subject='watch',primary_product='Rolex watch',secondary_products=[],objects=['watch','dial'],people=[],actions=[],environment='studio',shot_type='close_up',camera_motion='static',content_type='product_closeup',visual_attributes=['blue dial'],visual_concepts=['precision'],visible_text=['ROLEX'],observed_details=['blue dial visible'],uncertain_inferences=[],usable=True,analysis_confidence=.9,representative_paths=[],representative_times=[],analysis_hash='h')
    d.update(kw); return IndexedClip(**d)

def req(**kw):
    d=dict(segment_id='vs_1',start_word_id='w1',end_word_id='w3',narration_text='blue dial',visual_intent='literal_product_detail',requested_visual='close up blue Rolex dial',primary_subject='watch dial',required_entities=['Rolex'],required_attributes=['blue dial'],preferred_shot_types=['close_up'],preferred_content_types=['product_closeup'],visual_concepts=['dial design'],avoid=['human face visible'],specificity='high')
    d.update(kw); return VisualRequirement(**d)

class CoreTests(unittest.TestCase):
    def test_cosine(self): self.assertAlmostEqual(cosine([1,0],[1,0]),1)
    def test_lexical(self): self.assertGreater(lexical('blue rolex dial','rolex blue dial close up'),.4)
    def test_structured(self): self.assertGreater(structured(req(),clip()),.7)
    def test_clip_text_contains_semantics(self): self.assertIn('blue dial',clip_text(clip()))
    def test_usage_first_use(self): self.assertTrue(UsageLedger().can_use('c',0,False))
    def test_usage_no_repeat_without_flag(self):
        l=UsageLedger(); l.mark('c',0); self.assertFalse(l.can_use('c',4,False))
    def test_usage_gap(self):
        l=UsageLedger(min_gap=2); l.mark('c',0); self.assertFalse(l.can_use('c',2,True)); self.assertTrue(l.can_use('c',3,True))
    def test_usage_max_two(self):
        l=UsageLedger(); l.mark('c',0); l.mark('c',4); self.assertFalse(l.can_use('c',8,True))
    def test_source_window_second_use_differs(self):
        c=clip(); a=choose_window(c,4,1); b=choose_window(c,4,2); self.assertNotEqual(a,b)
    def test_extract_asr_words(self):
        data={'transcription':[{'tokens':[{'text':' Hello','offsets':{'from':100,'to':500}},{'text':' world','offsets':{'from':500,'to':900}}]}]}
        x=extract_asr_words(data); self.assertEqual([w['text'] for w in x],['Hello','world']); self.assertAlmostEqual(x[0]['start'],.1)
    def test_script_alignment_exact(self):
        data={'transcription':[{'tokens':[{'text':' Hello','offsets':{'from':100,'to':400}},{'text':' world','offsets':{'from':500,'to':900}}]}]}
        words,ratio=align_script('Hello world',data,1.0); self.assertEqual(ratio,1.0); self.assertTrue(all(w.aligned for w in words))
    def test_script_alignment_interpolates_missing(self):
        data={'transcription':[{'tokens':[{'text':' Hello','offsets':{'from':100,'to':300}},{'text':' world','offsets':{'from':800,'to':1000}}]}]}
        words,ratio=align_script('Hello beautiful world',data,1.2); self.assertEqual(len(words),3); self.assertGreater(words[1].start,words[0].end)

if __name__=='__main__': unittest.main()
