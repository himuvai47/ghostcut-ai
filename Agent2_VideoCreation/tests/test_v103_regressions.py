import sqlite3,tempfile,unittest
from pathlib import Path
from video_creation_agent.cache import RetrievalCache
from video_creation_agent.pipeline import unresolved_major_issues

class V103RegressionTests(unittest.TestCase):
    def test_cache_migrates_face_version_without_losing_embedding_schema(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'cache.db'
            c=sqlite3.connect(p)
            c.execute('CREATE TABLE clip_cache(clip_id TEXT PRIMARY KEY,analysis_hash TEXT NOT NULL,embed_model TEXT,embedding BLOB,face_status TEXT,face_reason TEXT,updated_at REAL NOT NULL)')
            c.commit(); c.close()
            cache=RetrievalCache(p)
            cols={r[1] for r in cache.conn.execute('PRAGMA table_info(clip_cache)')}
            self.assertIn('face_version',cols)
            cache.close()

    def test_resolved_major_review_issue_does_not_fail_qc(self):
        issues=[
            {'segment_id':'s1','severity':'major','action':'rematch','issue':'old mismatch','resolved':True,'resolution':'rematched'},
            {'segment_id':'s2','severity':'minor','action':'rematch','issue':'minor'},
        ]
        self.assertEqual(unresolved_major_issues(issues),[])
        issues.append({'segment_id':'s3','severity':'major','action':'rematch','issue':'still bad'})
        self.assertEqual(len(unresolved_major_issues(issues)),1)

if __name__=='__main__': unittest.main()
