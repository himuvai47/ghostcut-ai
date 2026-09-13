import json, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from video_creation_agent.models import Candidate, IndexedClip, VisualRequirement
from video_creation_agent.rerank import rerank
from video_creation_agent.config import Agent2Config


def clip(cid):
    return IndexedClip(
        clip_id=cid, source_id='s', source_video='v.mp4', source_path='v.mp4',
        start_time=0, end_time=5, duration=5,
        description='Close-up blue watch dial with silver bracelet',
        primary_subject='watch dial', primary_product='Rolex watch', secondary_products=[],
        objects=['watch'], people=[], actions=[], environment='studio', shot_type='close_up',
        camera_motion='static', content_type='product_closeup', visual_attributes=['blue dial'],
        visual_concepts=['watch design'], visible_text=['ROLEX'], observed_details=['blue dial visible'],
        uncertain_inferences=[], usable=True, analysis_confidence=.9,
        representative_paths=[], representative_times=[], analysis_hash='h'+cid,
    )


def req():
    return VisualRequirement(
        segment_id='vs1', start_word_id='w1', end_word_id='w2', narration_text='The blue dial stands out.',
        visual_intent='literal_product_detail', requested_visual='close-up blue Rolex dial',
        primary_subject='Rolex dial', required_entities=['Rolex'], required_attributes=['blue dial'],
        preferred_shot_types=['close_up'], preferred_content_types=['product_closeup'],
        visual_concepts=['dial design'], avoid=['face'], specificity='high',
    )


class TruncateThenRecover(BaseHTTPRequestHandler):
    calls = 0
    def do_POST(self):
        n=int(self.headers.get('Content-Length','0')); d=json.loads(self.rfile.read(n))
        TruncateThenRecover.calls += 1
        # Fail all compatibility modes for any request containing >1 candidate.
        text=d['messages'][-1].get('content','')
        ids=[]
        import re
        for cid in re.findall(r"'clip_id': '([^']+)'", text):
            if cid not in ids: ids.append(cid)
        if len(ids)>1:
            content='{"judgments":[{"clip_id":"'+ids[0]+'","reason":"unterminated'
        else:
            cid=ids[0]
            content=json.dumps({'judgments':[{'clip_id':cid,'subject_match':'strong','attribute_match':'strong','hard_requirement_failed':False,'decision':'strong_match','reason':'Good dial match.'}]})
        out={'message':{'content':content}}
        b=json.dumps(out).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def log_message(self,*a): pass


class V104RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=HTTPServer(('127.0.0.1',0),TruncateThenRecover)
        cls.t=threading.Thread(target=cls.s.serve_forever,daemon=True); cls.t.start()
    @classmethod
    def tearDownClass(cls): cls.s.shutdown()

    def test_reranker_splits_after_truncated_batch(self):
        cfg=Agent2Config(ollama_url=f'http://127.0.0.1:{self.s.server_port}')
        cs=[Candidate(clip(c),.8,.8,.8,.9,.8,face_status='no_face') for c in ['c1','c2','c3','c4']]
        out=rerank(cfg,req(),cs)
        self.assertEqual([x.clip_id for x in out],['c1','c2','c3','c4'])
        self.assertTrue(all(x.decision=='strong_match' for x in out))

    def test_single_candidate_runtime_failure_is_safe_reject(self):
        # Invalid URL guarantees the local reranker fails, but Agent 2 must not crash.
        cfg=Agent2Config(ollama_url='http://127.0.0.1:1')
        c=Candidate(clip('cx'),.8,.8,.8,.9,.8,face_status='uncertain')
        out=rerank(cfg,req(),[c])
        self.assertEqual(out[0].decision,'reject')
        self.assertIn('Local reranker',out[0].reason)

if __name__=='__main__': unittest.main()
