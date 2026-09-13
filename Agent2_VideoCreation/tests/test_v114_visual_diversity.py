import logging
import unittest

from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.usage import UsageLedger


def clip(cid, source, start, end, shot='close_up', content='product_closeup', desc='TAG Heuer watch'):
    return IndexedClip(
        clip_id=cid,source_id=source,source_video=f'{source}.mp4',source_path=f'{source}.mp4',
        start_time=float(start),end_time=float(end),duration=float(end-start),description=desc,
        primary_subject='watch',primary_product='TAG Heuer',secondary_products=[],objects=['watch'],
        people=[],actions=[],environment='studio',shot_type=shot,camera_motion='static',content_type=content,
        visual_attributes=[],visual_concepts=['watch'],visible_text=['TAG Heuer'],observed_details=[],
        uncertain_inferences=[],usable=True,analysis_confidence=.9,representative_paths=[],representative_times=[],
        analysis_hash='h'+cid,face_status='no_face',face_scan_version=2,face_scan_mode='representative_frames')


def cand(c, score):
    return Candidate(c,score,.1,.5,.9,score,face_status='no_face')


def req(specificity='normal'):
    return VisualRequirement(
        segment_id='vs',start_word_id='w1',end_word_id='w2',narration_text='TAG Heuer watch discussion',
        visual_intent='literal_product',requested_visual='TAG Heuer B-roll',primary_subject='TAG Heuer',
        required_entities=[],required_attributes=[],preferred_shot_types=[],preferred_content_types=[],
        visual_concepts=['watch'],avoid=[],specificity=specificity,timeline_start=0,timeline_end=6)


class V114VisualDiversityTests(unittest.TestCase):
    def pipeline(self):
        p=object.__new__(Agent2Pipeline)
        p.cfg=Agent2Config()
        p.log=logging.getLogger('v114')
        return p

    def test_nearby_same_source_clip_is_deprioritized(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        used=clip('used','s1',10,15)
        near=clip('near','s1',18,24)
        other=clip('other','s2',50,56)
        extras=[clip(f'e{i}',f's{i+3}',100+i*20,106+i*20) for i in range(6)]
        ledger.mark('used',0)
        clips=[used,near,other,*extras]
        ranked=[cand(near,.94),cand(other,.84),*[cand(c,.70-i*.01) for i,c in enumerate(extras)]]
        embs={c.clip_id:[0.0,1.0] for c in clips}
        out=p._apply_visual_diversity(ranked,clips,embs,ledger,1,req())
        self.assertNotEqual(out[0].clip.clip_id,'near')
        self.assertTrue(next(x for x in out if x.clip.clip_id=='near').near_duplicate)
        self.assertTrue(any('nearby_source_time' in x for x in next(x for x in out if x.clip.clip_id=='near').diversity_reasons))

    def test_embedding_near_duplicate_across_sources_is_deprioritized(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        used=clip('used','s1',0,6)
        dup=clip('dup','s2',50,56,desc='same product macro')
        different=clip('different','s3',70,76,shot='wide',content='environment',desc='watch boutique wide shot')
        extras=[clip(f'e{i}',f's{i+4}',100+i*20,106+i*20,shot='wide',content='environment') for i in range(6)]
        ledger.mark('used',0)
        clips=[used,dup,different,*extras]
        ranked=[cand(dup,.93),cand(different,.82),*[cand(c,.70) for c in extras]]
        embs={used.clip_id:[1.0,0.0],dup.clip_id:[1.0,0.0],different.clip_id:[0.0,1.0],**{c.clip_id:[0.0,1.0] for c in extras}}
        out=p._apply_visual_diversity(ranked,clips,embs,ledger,1,req())
        d=next(x for x in out if x.clip.clip_id=='dup')
        self.assertTrue(d.near_duplicate)
        self.assertTrue(any('near_duplicate_embedding' in x for x in d.diversity_reasons))
        self.assertNotEqual(out[0].clip.clip_id,'dup')

    def test_overused_source_loses_small_relevance_advantage(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        old=[]
        for i in range(4):
            c=clip(f'old{i}','s1',200+i*40,206+i*40); old.append(c); ledger.mark(c.clip_id,i)
        same=clip('same','s1',500,506)
        fresh=clip('fresh','s2',500,506)
        clips=[*old,same,fresh]
        ranked=[cand(same,.86),cand(fresh,.82)]
        embs={c.clip_id:[0.0,1.0] for c in clips}
        out=p._apply_visual_diversity(ranked,clips,embs,ledger,10,req())
        self.assertEqual(out[0].clip.clip_id,'fresh')
        self.assertGreater(next(x for x in out if x.clip.clip_id=='same').diversity_penalty,0)

    def test_diversity_metrics_expose_visual_repeat_signals(self):
        p=self.pipeline()
        a=clip('a','s1',0,6)
        b=clip('b','s1',8,14)
        c=clip('c','s2',30,36,shot='wide',content='environment')
        timeline=[
            {'clips':[{'clip_id':'a'}]},
            {'clips':[{'clip_id':'b'}]},
            {'clips':[{'clip_id':'c'}]},
        ]
        metrics=p._diversity_metrics(timeline,{x.clip_id:x for x in [a,b,c]}, {'a':[1,0],'b':[1,0],'c':[0,1]})
        self.assertEqual(metrics['same_source_consecutive_cuts'],1)
        self.assertGreaterEqual(metrics['nearby_source_region_reuses'],1)
        self.assertLess(metrics['visual_diversity_score'],1.0)
        self.assertEqual(metrics['source_distribution']['s1.mp4'],2)


if __name__=='__main__': unittest.main()
