from __future__ import annotations
import sqlite3, struct, time
from pathlib import Path

SCHEMA='''
CREATE TABLE IF NOT EXISTS clip_cache(
 clip_id TEXT PRIMARY KEY,
 analysis_hash TEXT NOT NULL,
 embed_model TEXT,
 embedding BLOB,
 face_status TEXT,
 face_reason TEXT,
 face_version TEXT,
 updated_at REAL NOT NULL
);
'''

class RetrievalCache:
 def __init__(self,path:Path):
  path.parent.mkdir(parents=True,exist_ok=True)
  self.root=path.parent
  self.conn=sqlite3.connect(path)
  self.conn.executescript(SCHEMA)
  cols={r[1] for r in self.conn.execute('PRAGMA table_info(clip_cache)').fetchall()}
  if 'face_version' not in cols:
   self.conn.execute('ALTER TABLE clip_cache ADD COLUMN face_version TEXT')
  self.conn.commit()
 def close(self): self.conn.close()
 def get_embedding(self,clip_id,analysis_hash,model):
  r=self.conn.execute('SELECT embedding FROM clip_cache WHERE clip_id=? AND analysis_hash=? AND embed_model=?',(clip_id,analysis_hash,model)).fetchone()
  if not r or r[0] is None:return None
  b=r[0]; return list(struct.unpack('<%df'%(len(b)//4),b))
 def put_embedding(self,clip_id,analysis_hash,model,vec):
  b=struct.pack('<%df'%len(vec),*vec)
  # Preserve face results only if they belong to the same source-analysis hash.
  old=self.conn.execute('SELECT analysis_hash,face_status,face_reason,face_version FROM clip_cache WHERE clip_id=?',(clip_id,)).fetchone()
  same=bool(old and old[0]==analysis_hash)
  fs,fr,fv=(old[1],old[2],old[3]) if same else (None,None,None)
  self.conn.execute('INSERT INTO clip_cache(clip_id,analysis_hash,embed_model,embedding,face_status,face_reason,face_version,updated_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(clip_id) DO UPDATE SET analysis_hash=excluded.analysis_hash,embed_model=excluded.embed_model,embedding=excluded.embedding,face_status=excluded.face_status,face_reason=excluded.face_reason,face_version=excluded.face_version,updated_at=excluded.updated_at',(clip_id,analysis_hash,model,b,fs,fr,fv,time.time()))
  self.conn.commit()
 def get_face(self,clip_id,analysis_hash,face_version=None):
  if face_version is None:
   r=self.conn.execute('SELECT face_status,face_reason FROM clip_cache WHERE clip_id=? AND analysis_hash=?',(clip_id,analysis_hash)).fetchone()
  else:
   r=self.conn.execute('SELECT face_status,face_reason FROM clip_cache WHERE clip_id=? AND analysis_hash=? AND face_version=?',(clip_id,analysis_hash,face_version)).fetchone()
  return tuple(r) if r and r[0] else None
 def put_face(self,clip_id,analysis_hash,status,reason,face_version=None):
  self.conn.execute('INSERT INTO clip_cache(clip_id,analysis_hash,updated_at,face_status,face_reason,face_version) VALUES(?,?,?,?,?,?) ON CONFLICT(clip_id) DO UPDATE SET analysis_hash=excluded.analysis_hash,face_status=excluded.face_status,face_reason=excluded.face_reason,face_version=excluded.face_version,updated_at=excluded.updated_at',(clip_id,analysis_hash,time.time(),status,reason,face_version))
  self.conn.commit()
