import json, sqlite3, tempfile, unittest
from pathlib import Path
from video_creation_agent.agent1_reader import load_agent1_clips

SCHEMA = """CREATE TABLE sources(id INTEGER PRIMARY KEY,source_uid TEXT,relative_path TEXT,absolute_path TEXT,fingerprint TEXT);
CREATE TABLE shots(id INTEGER PRIMARY KEY,shot_uid TEXT,source_id INTEGER);
CREATE TABLE clips(id INTEGER PRIMARY KEY,clip_uid TEXT,source_id INTEGER,shot_id INTEGER,clip_index INTEGER,start_time REAL,end_time REAL,duration REAL,representative_paths_json TEXT,representative_times_json TEXT,analysis_json TEXT,description TEXT,usable INTEGER,analysis_confidence REAL,status TEXT,face_status TEXT,face_scan_version INTEGER,face_scan_mode TEXT);"""
OLD_SCHEMA = """CREATE TABLE sources(id INTEGER PRIMARY KEY,source_uid TEXT,relative_path TEXT,absolute_path TEXT,fingerprint TEXT);
CREATE TABLE shots(id INTEGER PRIMARY KEY,shot_uid TEXT,source_id INTEGER);
CREATE TABLE clips(id INTEGER PRIMARY KEY,clip_uid TEXT,source_id INTEGER,shot_id INTEGER,clip_index INTEGER,start_time REAL,end_time REAL,duration REAL,representative_paths_json TEXT,representative_times_json TEXT,analysis_json TEXT,description TEXT,usable INTEGER,analysis_confidence REAL,status TEXT);"""

def add_base(c, schema=SCHEMA, status='no_face', version=2):
    c.executescript(schema)
    c.execute("INSERT INTO sources VALUES(1,'s1','v.mp4','C:/v.mp4','fp1')")
    c.execute("INSERT INTO shots VALUES(1,'sh1',1)")
    a={'description':'watch','primary_subject':'watch','primary_product':'Rolex','secondary_products':[],'objects':['watch'],'people':[],'actions':[],'environment':'studio','shot_type':'close_up','camera_motion':'static','content_type':'product_closeup','visual_attributes':[],'visual_concepts':[],'visible_text':['ROLEX'],'observed_details':[],'uncertain_inferences':[],'usable':True,'analysis_confidence':.88}
    if schema == SCHEMA:
        c.execute("INSERT INTO clips VALUES(1,'c1',1,1,1,0,5,5,'[]','[]',?,?,1,.88,'INDEXED',?,?,?)",(json.dumps(a),'watch',status,version,'representative_frames'))
    else:
        c.execute("INSERT INTO clips VALUES(1,'c1',1,1,1,0,5,5,'[]','[]',?,?,1,.88,'INDEXED')",(json.dumps(a),'watch'))
    c.commit()

class ReaderTests(unittest.TestCase):
    def test_load_persistent_face_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'footage_index.db'; c=sqlite3.connect(p); add_base(c); c.close()
            xs=load_agent1_clips(Path(td))
            self.assertEqual(len(xs),1); self.assertEqual(xs[0].primary_product,'Rolex')
            self.assertEqual(xs[0].face_status,'no_face'); self.assertEqual(xs[0].face_scan_version,2)

    def test_old_agent1_schema_requires_backfill_upgrade(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'footage_index.db'; c=sqlite3.connect(p); add_base(c,OLD_SCHEMA); c.close()
            with self.assertRaisesRegex(RuntimeError,'FaceBackfill'):
                load_agent1_clips(Path(td))

    def test_stale_face_scan_version_requires_backfill(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'footage_index.db'; c=sqlite3.connect(p); add_base(c,status='uncertain',version=1); c.close()
            with self.assertRaisesRegex(RuntimeError,'FaceBackfill'):
                load_agent1_clips(Path(td))

    def test_uncertain_v2_is_valid_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'footage_index.db'; c=sqlite3.connect(p); add_base(c,status='uncertain',version=2); c.close()
            xs=load_agent1_clips(Path(td)); self.assertEqual(xs[0].face_status,'uncertain')

if __name__=='__main__': unittest.main()
