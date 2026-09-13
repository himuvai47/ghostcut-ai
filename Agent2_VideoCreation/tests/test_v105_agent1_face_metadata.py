import unittest
from unittest.mock import patch
from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, IndexedClip, VisualRequirement, CandidateJudgment
from video_creation_agent.pipeline import Agent2Pipeline


def clip(cid,status):
    return IndexedClip(
        clip_id=cid, source_id='s', source_video='v.mp4', source_path='v.mp4', start_time=0,end_time=5,duration=5,
        description='watch',primary_subject='watch',primary_product='watch',secondary_products=[],objects=['watch'],people=[],actions=[],environment='studio',shot_type='close_up',camera_motion='static',content_type='product_closeup',visual_attributes=[],visual_concepts=[],visible_text=[],observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,representative_paths=[],representative_times=[],analysis_hash='h'+cid,face_status=status,face_scan_version=2,face_scan_mode='representative_frames')

class DummyCache:
    def close(self): pass

class V105Tests(unittest.TestCase):
    def pipeline(self,allow=True):
        p=object.__new__(Agent2Pipeline); p.cfg=Agent2Config(allow_uncertain_face_status=allow); p.cache=DummyCache(); return p

    def test_shortlist_uses_agent1_metadata_without_scanner(self):
        p=self.pipeline(True)
        ranked=[Candidate(clip('a','no_face'),.8,.1,.5,.9,.8),Candidate(clip('b','uncertain'),.7,.1,.5,.9,.7),Candidate(clip('c','face_visible'),.9,.1,.5,.9,.9)]
        accepted,checked=p._face_free_shortlist(ranked)
        self.assertEqual([x.clip.clip_id for x in accepted],['a','b'])
        self.assertEqual([x.face_status for x in checked],['no_face','uncertain','face_visible'])
        self.assertTrue(all('Agent 1 persistent' in x.face_reason for x in checked))

    def test_uncertain_can_be_disabled_without_runtime_rescan(self):
        p=self.pipeline(False)
        ranked=[Candidate(clip('b','uncertain'),.7,.1,.5,.9,.7)]
        accepted,checked=p._face_free_shortlist(ranked)
        self.assertEqual(accepted,[]); self.assertEqual(checked[0].face_status,'uncertain')


    def test_progressive_rerank_stops_after_first_sufficient_batch(self):
        p=self.pipeline(True)
        r=VisualRequirement(segment_id='vs',start_word_id='w1',end_word_id='w2',narration_text='watch',visual_intent='literal_product',requested_visual='watch close-up',primary_subject='watch',specificity='normal',timeline_start=0,timeline_end=3)
        pool=[Candidate(clip(str(i),'no_face'),.8,.1,.5,.9,.8) for i in range(8)]
        def fake(_cfg,_req,batch):
            return [CandidateJudgment(c.clip.clip_id,'strong','strong','strong','strong',False,[],'strong_match','good') for c in batch]
        with patch('video_creation_agent.pipeline.rerank',side_effect=fake) as rr:
            out=p._judge_pool(r,pool,target_duration=3,allow_weak=True)
        self.assertEqual(rr.call_count,1)
        self.assertEqual(len(out),2)

    def test_legacy_face_runtime_settings_no_longer_change_config_signature(self):
        a=Agent2Config(face_model='qwen2.5vl:7b',face_scan_max_frames=16,face_frame_width=768)
        b=Agent2Config(face_model='some-other-vision-model',face_scan_max_frames=99,face_frame_width=2048)
        self.assertEqual(a.signature(),b.signature())

if __name__=='__main__': unittest.main()
