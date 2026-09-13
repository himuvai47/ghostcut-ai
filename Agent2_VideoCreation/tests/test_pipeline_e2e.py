import hashlib,json,sqlite3,tempfile,threading,unittest,re
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from video_creation_agent.config import Agent2Config
from video_creation_agent.pipeline import Agent2Pipeline
from video_creation_agent.utils import write_json,sha256_file
import logging

AG1='''
CREATE TABLE sources(id INTEGER PRIMARY KEY,source_uid TEXT,relative_path TEXT,absolute_path TEXT,fingerprint TEXT);
CREATE TABLE shots(id INTEGER PRIMARY KEY,shot_uid TEXT,source_id INTEGER);
CREATE TABLE clips(id INTEGER PRIMARY KEY,clip_uid TEXT,source_id INTEGER,shot_id INTEGER,clip_index INTEGER,start_time REAL,end_time REAL,duration REAL,representative_paths_json TEXT,representative_times_json TEXT,analysis_json TEXT,description TEXT,usable INTEGER,analysis_confidence REAL,status TEXT,face_status TEXT,face_scan_version INTEGER,face_scan_mode TEXT);
'''
class H(BaseHTTPRequestHandler):
 ranking_calls=0
 review_calls=0
 face_calls=0
 def do_POST(self):
  n=int(self.headers.get('Content-Length','0')); d=json.loads(self.rfile.read(n))
  if self.path=='/api/embed':
   ins=d['input']; ins=[ins] if isinstance(ins,str) else ins
   def v(x):
    x=x.lower(); return [1.0,0.0,0.1] if 'rolex' in x or 'blue dial' in x else [0.1,1.0,0.0]
   out={'embeddings':[v(x) for x in ins]}
  elif self.path=='/api/chat':
   text=d['messages'][-1].get('content','')
   if 'HARD faceless-video gate' in text or 'HARD gate for a faceless' in text:
    H.face_calls += 1
    raise AssertionError('Agent 2 v1.0.5 must not perform runtime face screening')
   elif 'rank indexed footage for one' in text.lower():
    H.ranking_calls += 1
    ids=[]
    for cid in re.findall(r"'clip_id': '([^']+)'",text):
     if cid not in ids: ids.append(cid)
    content={'judgments':[{'clip_id':cid,'subject_match':'strong','attribute_match':'strong','hard_requirement_failed':False,'decision':'strong_match','reason':'Mock relevant product-detail match.'} for cid in ids]}
   elif 'review this faceless rough-cut' in text.lower():
    H.review_calls += 1
    content={'issues':[]}
   else: raise AssertionError('unexpected chat prompt')
   out={'message':{'content':json.dumps(content)}}
  else: self.send_response(404); self.end_headers(); return
  b=json.dumps(out).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
 def log_message(self,*a):pass

class E2E(unittest.TestCase):
 @classmethod
 def setUpClass(cls): cls.s=HTTPServer(('127.0.0.1',0),H); cls.t=threading.Thread(target=cls.s.serve_forever,daemon=True); cls.t.start()
 @classmethod
 def tearDownClass(cls): cls.s.shutdown()
 def test_pipeline_rejects_face_and_builds_timeline(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td); a1=root/'a1'; a1.mkdir(); ws=root/'a2'; script=root/'script.txt'; audio=root/'voice.wav'
   script.write_text('The blue Rolex dial is easy to recognize.',encoding='utf-8'); audio.write_bytes(b'audio')
   noface=root/'noface.jpg'; noface.write_bytes(b'CLEAR_IMAGE'); face=root/'face.jpg'; face.write_bytes(b'FACE_PRESENT')
   (a1/'qc_report.json').write_text(json.dumps({'passed':True}),encoding='utf-8'); (root/'rolex.mp4').write_bytes(b'video'); db=sqlite3.connect(a1/'footage_index.db'); db.executescript(AG1)
   db.execute("INSERT INTO sources VALUES(1,'s1','rolex.mp4',?,'fp1')",(str(root/'rolex.mp4'),)); db.execute("INSERT INTO shots VALUES(1,'sh1',1)")
   for i,(cid,desc,img,face_status) in enumerate([('clip_face','blue Rolex dial very close face',[str(face)],'face_visible'),('clip_clean','blue Rolex dial product close-up',[str(noface)],'no_face')],1):
    an={'description':desc,'primary_subject':'Rolex watch','primary_product':'Rolex Oyster Perpetual','secondary_products':[],'objects':['watch','dial'],'people':[],'actions':[],'environment':'studio','shot_type':'close_up','camera_motion':'static','content_type':'product_closeup','visual_attributes':['blue dial'],'visual_concepts':['dial design'],'visible_text':['ROLEX'],'observed_details':['blue dial visible'],'uncertain_inferences':[],'usable':True,'analysis_confidence':.9}
    db.execute("INSERT INTO clips VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(i,cid,1,1,i,0,5,5,json.dumps(img),json.dumps([1.0]),json.dumps(an),desc,1,.9,'INDEXED',face_status,2,'representative_frames'))
   db.commit(); db.close()
   cfg=Agent2Config(ollama_url=f'http://127.0.0.1:{self.s.server_port}',global_review_enabled=True,retrieval_top_k=10,rerank_top_k=4)
   # Preseed alignment + visual segments so test isolates downstream Agent 2 pipeline without whisper binary.
   ws.mkdir(); align={'audio_duration':4.0,'alignment_ratio':1.0,'words':[{'id':f'w{i+1:05d}','text':w,'start':i*.5,'end':(i+1)*.5,'aligned':True,'asr_text':w} for i,w in enumerate(['The','blue','Rolex','dial','is','easy','to','recognize'])]}; write_json(ws/'alignment.json',align)
   seg={'segment_id':'vs_0001','start_word_id':'w00001','end_word_id':'w00008','narration_text':'The blue Rolex dial is easy to recognize','visual_intent':'literal_product_detail','requested_visual':'close-up blue Rolex dial','primary_subject':'Rolex dial','required_entities':['Rolex'],'required_attributes':['blue dial'],'preferred_shot_types':['close_up'],'preferred_content_types':['product_closeup'],'visual_concepts':['dial design'],'avoid':['human face visible'],'specificity':'high','speech_start':0.0,'speech_end':4.0,'timeline_start':0.0,'timeline_end':4.0}; write_json(ws/'visual_segments.json',{'segments':[seg]})
   from video_creation_agent.planner import PLANNER_VERSION
   state={'script_sha256':sha256_file(script),'audio_sha256':sha256_file(audio),'planner_version':PLANNER_VERSION,'visual_segments_ready':True}; write_json(ws/'state.json',state)
   log=logging.getLogger('e2e'); log.handlers.clear(); log.addHandler(logging.NullHandler())
   p=Agent2Pipeline(cfg,ws,Path('/nonexistent'),log)
   try: summary=p.run(script,audio,a1,False,False)
   finally:p.close()
   self.assertTrue(summary['qc_passed']); tl=json.loads((ws/'timeline.json').read_text()); self.assertEqual(tl['segments'][0]['clips'][0]['clip_id'],'clip_clean'); self.assertEqual(tl['segments'][0]['clips'][0]['face_status'],'no_face'); self.assertEqual(H.face_calls,0); self.assertEqual(H.review_calls,0)
   first_calls=H.ranking_calls
   p2=Agent2Pipeline(cfg,ws,Path('/nonexistent'),log)
   try: summary2=p2.run(script,audio,a1,False,False)
   finally:p2.close()
   self.assertTrue(summary2['qc_passed']); self.assertEqual(H.ranking_calls,first_calls)

if __name__=='__main__':unittest.main()
