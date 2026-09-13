import base64, json, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from video_creation_agent.cache import RetrievalCache
from video_creation_agent.config import Agent2Config
from video_creation_agent.face_gate import screen_clip
from video_creation_agent.models import IndexedClip
from video_creation_agent.ollama import embed, chat

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0')); body=json.loads(self.rfile.read(n))
        if self.path=='/api/embed':
            ins=body['input']; ins=[ins] if isinstance(ins,str) else ins
            out={'embeddings':[[float(len(x)),1.0,0.5] for x in ins]}
        elif self.path=='/api/chat':
            msg=body['messages'][-1]
            status='no_face'
            for im in msg.get('images',[]):
                raw=base64.b64decode(im)
                if b'FACE' in raw: status='face_visible'
            out={'message':{'content':json.dumps({'status':status,'reason':'test'})}}
        else:
            self.send_response(404); self.end_headers(); return
        b=json.dumps(out).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self,*a): pass

class HttpTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.srv=HTTPServer(('127.0.0.1',0),Handler); cls.port=cls.srv.server_port; cls.t=threading.Thread(target=cls.srv.serve_forever,daemon=True); cls.t.start()
 @classmethod
 def tearDownClass(cls): cls.srv.shutdown()
 def test_embed(self): self.assertEqual(len(embed(f'http://127.0.0.1:{self.port}','m',['a','bb'])),2)
 def test_face_gate_and_cache(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'f.jpg'; p.write_bytes(b'NOFACE')
   # avoid substring FACE in no-face fixture
   p.write_bytes(b'CLEAR_IMAGE')
   c=IndexedClip('c','s','v','v',0,1,1,'d','p','none',[],[],[],[], '','unknown','unknown','other',[],[],[],[],[],True,.8,[str(p)],[.5],'h')
   cache=RetrievalCache(Path(td)/'c.db'); cfg=Agent2Config(ollama_url=f'http://127.0.0.1:{self.port}')
   self.assertEqual(screen_clip(cfg,c,cache)[0],'no_face'); self.assertEqual(screen_clip(cfg,c,cache)[0],'no_face'); cache.close()
 def test_face_visible(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'f.jpg'; p.write_bytes(b'FACE')
   c=IndexedClip('c','s','v','v',0,1,1,'d','p','none',[],[],[],[], '','unknown','unknown','other',[],[],[],[],[],True,.8,[str(p)],[.5],'h')
   cache=RetrievalCache(Path(td)/'c.db'); cfg=Agent2Config(ollama_url=f'http://127.0.0.1:{self.port}')
   self.assertEqual(screen_clip(cfg,c,cache)[0],'face_visible'); cache.close()

if __name__=='__main__': unittest.main()
