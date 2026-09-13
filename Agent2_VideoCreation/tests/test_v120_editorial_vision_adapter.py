import json
import logging
import sqlite3
import tempfile
import unittest
from pathlib import Path

from video_creation_agent.agent1_reader import load_agent1_clips
from video_creation_agent.config import Agent2Config
from video_creation_agent.models import Candidate, IndexedClip, VisualRequirement
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.usage import UsageLedger


def clip(cid, source, *, score=.9, family='unknown', role='other', focus='unknown', brand='unknown', model='unknown', profile=0,
         quality=.5, usefulness=.5, start=0.0, shot='close_up', content='product_closeup'):
    return IndexedClip(
        clip_id=cid,source_id=source,source_video=f'{source}.mp4',source_path=f'{source}.mp4',
        start_time=float(start),end_time=float(start+6),duration=6.0,description='TAG Heuer watch footage',
        primary_subject='watch',primary_product='TAG Heuer watch',secondary_products=[],objects=['watch'],people=[],actions=[],
        environment='studio',shot_type=shot,camera_motion='static',content_type=content,visual_attributes=[],visual_concepts=['watch'],
        visible_text=['TAG Heuer'],observed_details=[],uncertain_inferences=[],usable=True,analysis_confidence=.9,
        representative_paths=[],representative_times=[],analysis_hash='h'+cid,face_status='no_face',face_scan_version=2,
        face_scan_mode='representative_frames',brand=brand,model_name=model,subject_focus=focus,shot_angle='front',
        editorial_role=role,visual_energy='calm',lighting_style='neutral',composition='single_subject',visual_family=family,
        visual_quality_score=quality,editorial_usefulness_score=usefulness,editorial_profile_version=profile,
    )


def cand(c, score):
    return Candidate(c, score, .1, .5, .9, score, face_status='no_face')


def req(mode='thematic', text='TAG Heuer watch discussion'):
    return VisualRequirement(
        segment_id='vs',start_word_id='w1',end_word_id='w2',narration_text=text,
        visual_intent='conceptual',requested_visual='Relevant TAG Heuer B-roll',primary_subject='TAG Heuer',
        required_entities=[],required_attributes=[],preferred_shot_types=[],preferred_content_types=[],visual_concepts=['watch'],
        avoid=[],specificity='normal',match_mode=mode,context_anchor='TAG Heuer',timeline_start=0,timeline_end=6,
    )


class V120EditorialVisionAdapterTests(unittest.TestCase):
    def pipeline(self):
        p=object.__new__(Agent2Pipeline)
        p.cfg=Agent2Config()
        p.log=logging.getLogger('v120')
        return p

    def test_exact_agent1_visual_family_reuse_is_deprioritized(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        used=clip('used','s1',family='tag_heuer_carrera__technical_detail__dial__close_up__front__bright',role='technical_detail',focus='dial',brand='TAG Heuer',model='Carrera',profile=1,start=0)
        duplicate=clip('dup','s2',family=used.visual_family,role='technical_detail',focus='dial',brand='TAG Heuer',model='Carrera',profile=1,start=100)
        fresh=clip('fresh','s3',family='tag_heuer_carrera__product_beauty__full_product__medium__three_quarter__dark',role='product_beauty',focus='full_product',brand='TAG Heuer',model='Carrera',profile=1,start=100)
        ledger.mark('used',0)
        ranked=[cand(duplicate,.94),cand(fresh,.84)]
        embs={c.clip_id:[0.0,1.0] for c in [used,duplicate,fresh]}
        out=p._apply_visual_diversity(ranked,[used,duplicate,fresh],embs,ledger,1,req())
        d=next(x for x in out if x.clip.clip_id=='dup')
        self.assertTrue(d.near_duplicate)
        self.assertIn('agent1_visual_family_recent',d.diversity_reasons)
        self.assertEqual(out[0].clip.clip_id,'fresh')

    def test_same_role_and_focus_are_softly_rotated(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        used=clip('used','s1',family='f1',role='technical_detail',focus='dial',brand='TAG Heuer',profile=1,start=0)
        same=clip('same','s2',family='f2',role='technical_detail',focus='dial',brand='TAG Heuer',profile=1,start=100)
        varied=clip('varied','s3',family='f3',role='product_beauty',focus='full_product',brand='TAG Heuer',profile=1,start=100)
        ledger.mark('used',0)
        out=p._apply_visual_diversity([cand(same,.87),cand(varied,.84)],[used,same,varied],{c.clip_id:[0,1] for c in [used,same,varied]},ledger,1,req())
        s=next(x for x in out if x.clip.clip_id=='same')
        self.assertIn('agent1_same_role_focus_recent',s.diversity_reasons)
        self.assertEqual(out[0].clip.clip_id,'varied')

    def test_high_quality_profiled_clip_gets_positive_tiebreaker(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        enhanced=clip('enh','s1',family='f1',role='product_beauty',focus='full_product',brand='TAG Heuer',model='Carrera',profile=1,quality=.95,usefulness=.98,start=100)
        legacy=clip('legacy','s2',profile=0,start=100)
        out=p._apply_visual_diversity([cand(legacy,.86),cand(enhanced,.83)],[legacy,enhanced],{'legacy':[0,1],'enh':[0,1]},ledger,5,req('brand_filler'))
        e=next(x for x in out if x.clip.clip_id=='enh')
        self.assertGreater(e.editorial_bonus,0)
        self.assertTrue(any(x.startswith('editorial_quality:') for x in e.editorial_reasons))
        self.assertEqual(out[0].clip.clip_id,'enh')

    def test_legacy_clip_keeps_zero_editorial_adjustment(self):
        p=self.pipeline(); ledger=UsageLedger(2,2)
        a=clip('a','s1',profile=0,start=100); b=clip('b','s2',profile=0,start=100)
        out=p._apply_visual_diversity([cand(a,.88),cand(b,.84)],[a,b],{'a':[0,1],'b':[0,1]},ledger,5,req())
        self.assertEqual(out[0].clip.clip_id,'a')
        self.assertTrue(all(x.editorial_bonus==0 for x in out))
        self.assertTrue(all(x.editorial_reasons==[] for x in out))

    def test_profiled_clip_gets_new_embedding_cache_namespace_only(self):
        p=self.pipeline()
        legacy=clip('legacy','s1',profile=0)
        enhanced=clip('enh','s2',family='family_x',role='product_beauty',focus='full_product',brand='TAG Heuer',profile=1)
        self.assertEqual(p._clip_embedding_cache_hash(legacy),legacy.analysis_hash)
        self.assertNotEqual(p._clip_embedding_cache_hash(enhanced),enhanced.analysis_hash)

    def test_diversity_metrics_report_editorial_family_signals(self):
        p=self.pipeline()
        a=clip('a','s1',family='family_a',role='technical_detail',focus='dial',brand='TAG Heuer',profile=1,start=0)
        b=clip('b','s2',family='family_a',role='technical_detail',focus='dial',brand='TAG Heuer',profile=1,start=100)
        c=clip('c','s3',family='family_c',role='product_beauty',focus='full_product',brand='TAG Heuer',profile=1,start=200)
        timeline=[{'clips':[{'clip_id':'a'}]},{'clips':[{'clip_id':'b'}]},{'clips':[{'clip_id':'c'}]}]
        metrics=p._diversity_metrics(timeline,{x.clip_id:x for x in [a,b,c]},{'a':[1,0],'b':[0,1],'c':[-1,0]})
        self.assertEqual(metrics['visual_family_reuses'],1)
        self.assertEqual(metrics['same_editorial_role_focus_consecutive'],1)
        self.assertEqual(metrics['agent1_editorial_profile_clips_used'],3)
        self.assertEqual(metrics['unique_visual_families_used'],2)

    def test_reader_loads_v13_metadata_and_leaves_legacy_defaults(self):
        schema="""CREATE TABLE sources(id INTEGER PRIMARY KEY,source_uid TEXT,relative_path TEXT,absolute_path TEXT,fingerprint TEXT);\nCREATE TABLE shots(id INTEGER PRIMARY KEY,shot_uid TEXT,source_id INTEGER);\nCREATE TABLE clips(id INTEGER PRIMARY KEY,clip_uid TEXT,source_id INTEGER,shot_id INTEGER,clip_index INTEGER,start_time REAL,end_time REAL,duration REAL,representative_paths_json TEXT,representative_times_json TEXT,analysis_json TEXT,description TEXT,usable INTEGER,analysis_confidence REAL,status TEXT,face_status TEXT,face_scan_version INTEGER,face_scan_mode TEXT);"""
        with tempfile.TemporaryDirectory() as td:
            db=Path(td)/'footage_index.db'; conn=sqlite3.connect(db); conn.executescript(schema)
            conn.execute("INSERT INTO sources VALUES(1,'s1','v.mp4','C:/v.mp4','fp')")
            conn.execute("INSERT INTO shots VALUES(1,'sh1',1)")
            base={'description':'watch','primary_subject':'watch','primary_product':'TAG Heuer','secondary_products':[],'objects':['watch'],'people':[],'actions':[],'environment':'studio','shot_type':'close_up','camera_motion':'static','content_type':'product_closeup','visual_attributes':[],'visual_concepts':[],'visible_text':['TAG Heuer'],'observed_details':[],'uncertain_inferences':[],'usable':True,'analysis_confidence':.9}
            enhanced=dict(base,brand='TAG Heuer',model_name='Carrera',subject_focus='dial',shot_angle='front',editorial_role='technical_detail',visual_energy='calm',lighting_style='bright',composition='single_subject',visual_family='family_a',visual_quality_score=.91,editorial_usefulness_score=.97,editorial_profile_version=1)
            conn.execute("INSERT INTO clips VALUES(1,'c1',1,1,1,0,5,5,'[]','[]',?,?,1,.9,'INDEXED','no_face',2,'representative_frames')",(json.dumps(enhanced),'watch'))
            conn.execute("INSERT INTO clips VALUES(2,'c2',1,1,2,6,11,5,'[]','[]',?,?,1,.9,'INDEXED','no_face',2,'representative_frames')",(json.dumps(base),'watch'))
            conn.commit(); conn.close()
            xs=load_agent1_clips(Path(td)); by={x.clip_id:x for x in xs}
            self.assertTrue(by['c1'].has_editorial_profile)
            self.assertEqual(by['c1'].brand,'TAG Heuer'); self.assertEqual(by['c1'].visual_family,'family_a')
            self.assertAlmostEqual(by['c1'].editorial_usefulness_score,.97)
            self.assertFalse(by['c2'].has_editorial_profile)
            self.assertEqual(by['c2'].visual_family,'unknown')


if __name__=='__main__':
    unittest.main()
