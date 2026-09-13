import json, logging, shutil, subprocess, tempfile, unittest
from pathlib import Path
from renderer_agent.config import Agent3Config
from renderer_agent.pipeline import Agent3Pipeline
from renderer_agent.probe import probe, duration_seconds, video_stream, audio_stream, parse_rate

def cmd(args): subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
class IntegrationTests(unittest.TestCase):
    def make_media(self,d:Path):
        v1=d/'v1.mp4'; v2=d/'v2.mp4'; a=d/'voice.wav'
        cmd(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc2=s=320x240:r=30:d=3','-an','-c:v','libx264','-pix_fmt','yuv420p',str(v1)])
        cmd(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc=s=240x320:r=30:d=3','-an','-c:v','libx264','-pix_fmt','yuv420p',str(v2)])
        cmd(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=6','-c:a','pcm_s16le',str(a)])
        return v1,v2,a
    def test_end_to_end_and_cache(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); v1,v2,a=self.make_media(d); ws=d/'out'; tp=d/'timeline.json'
            timeline={'schema_version':1,'audio':{'source':str(a),'duration':6.0},'segments':[
                {'segment_id':'s1','status':'matched','clips':[{'clip_id':'c1','source':str(v1),'source_start':0,'source_end':3,'timeline_start':0,'timeline_end':3}]},
                {'segment_id':'s2','status':'matched','clips':[{'clip_id':'c2','source':str(v2),'source_start':0,'source_end':3,'timeline_start':3,'timeline_end':6}]}]}
            tp.write_text(json.dumps(timeline))
            log=logging.getLogger('itest'); log.handlers.clear(); log.addHandler(logging.NullHandler())
            cfg=Agent3Config(width=640,height=360,fps=30,encoder='libx264',x264_preset='ultrafast')
            s1=Agent3Pipeline(cfg,ws,log).run(tp); self.assertTrue(s1['qc_passed']); self.assertEqual(s1['clips_rendered'],2)
            info=probe(ws/'rough_cut.mp4'); self.assertIsNotNone(video_stream(info)); self.assertIsNotNone(audio_stream(info)); self.assertAlmostEqual(duration_seconds(info),6.0,delta=.1)
            vs=video_stream(info); self.assertEqual((vs['width'],vs['height']),(640,360)); self.assertAlmostEqual(parse_rate(vs['avg_frame_rate']),30,delta=.05)
            s2=Agent3Pipeline(cfg,ws,log).run(tp); self.assertEqual(s2['clips_cached'],2)
    def test_gap_freeze_fill(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td); v1,v2,a=self.make_media(d); ws=d/'out'; tp=d/'timeline.json'
            timeline={'audio':{'source':str(a),'duration':6.0},'segments':[
                {'segment_id':'s1','status':'matched','clips':[{'clip_id':'c1','source':str(v1),'source_start':0,'source_end':3,'timeline_start':0,'timeline_end':3}]},
                {'segment_id':'s2','status':'matched','clips':[{'clip_id':'c2','source':str(v2),'source_start':0,'source_end':2,'timeline_start':4,'timeline_end':6}]}]}
            tp.write_text(json.dumps(timeline)); log=logging.getLogger('gap'); log.handlers.clear(); log.addHandler(logging.NullHandler())
            cfg=Agent3Config(width=320,height=180,fps=30,encoder='libx264',x264_preset='ultrafast')
            s=Agent3Pipeline(cfg,ws,log).run(tp); self.assertTrue(s['qc_passed']); self.assertEqual(s['gap_fillers'],1)

if __name__=='__main__': unittest.main()
