from __future__ import annotations
import logging
from pathlib import Path
from .config import Agent3Config
from .timeline_reader import load_timeline
from .preflight import run_preflight
from .encoder import choose_encoder
from .render_plan import build_render_plan
from .clip_renderer import render_clip, render_freeze_gap
from .assemble import concatenate, mux_voiceover
from .qc import validate_output
from .utils import write_json

class Agent3Pipeline:
    def __init__(self,cfg:Agent3Config,workspace:Path,logger:logging.Logger):
        self.cfg=cfg; self.ws=workspace; self.log=logger

    def run(self,timeline_path:Path,force:bool=False)->dict:
        self.ws.mkdir(parents=True,exist_ok=True)
        timeline=load_timeline(timeline_path)
        pf=run_preflight(timeline,self.ws)
        encoder=choose_encoder(self.cfg)
        plan=build_render_plan(timeline,self.cfg,encoder)
        write_json(self.ws/'render_plan.json',plan)
        self.log.info(f"Timeline: {len(timeline['segments'])} segments / {sum(1 for i in plan['items'] if i['type']=='clip')} selected clips")
        self.log.info(f"Target duration: {plan['target_duration']:.3f} sec")
        self.log.info(f"Output: {self.cfg.width}x{self.cfg.height} @ {self.cfg.fps}fps")
        self.log.info(f"Encoder: {encoder}")
        cache_dir=self.ws/'cache'/'clips'; rendered=[]; clips_rendered=0; clips_cached=0; gap_count=0
        # Render selected clips first so gap fillers can freeze an adjacent normalized frame.
        clip_outputs=[]
        clip_items=[i for i in plan['items'] if i['type']=='clip']
        for n,item in enumerate(clip_items,1):
            self.log.info(f"Rendering clip {n}/{len(clip_items)}: {item['clip_id']}")
            p,cached=render_clip(item,self.cfg,encoder,cache_dir,force=force)
            clip_outputs.append((item,p)); clips_cached+=int(cached); clips_rendered+=int(not cached)
            if cached: self.log.info(f"Clip {n}/{len(clip_items)} cached -> reuse")
        path_by_key={(i['start_frame'],i['end_frame']):p for i,p in clip_outputs}
        items=plan['items']
        for idx,item in enumerate(items):
            if item['type']=='clip':
                rendered.append(path_by_key[(item['start_frame'],item['end_frame'])]); continue
            gap_count+=1
            if item['duration']>=self.cfg.gap_warning_seconds:
                self.log.warning(f"Timeline gap {item['duration']:.3f}s -> freeze-frame filler")
            prev=rendered[-1] if rendered else None
            nxt=None
            if not prev:
                for j in range(idx+1,len(items)):
                    if items[j]['type']=='clip':
                        nxt=path_by_key[(items[j]['start_frame'],items[j]['end_frame'])]; break
            src=prev or nxt
            if src is None: raise RuntimeError("Cannot create gap filler: no adjacent video clip exists")
            gp=self.ws/'cache'/'gaps'/f"gap_{item['start_frame']:09d}_{item['end_frame']:09d}.mp4"
            if force or not gp.exists(): render_freeze_gap(src,item,self.cfg,encoder,gp,use_first_frame=(prev is None))
            rendered.append(gp)
        assembled=self.ws/'assembled_video.mp4'
        concatenate(rendered,self.ws/'manifests'/'concat.ffconcat',assembled)
        self.log.info(f"Concatenating {len(rendered)} timeline item(s)")
        final=self.ws/'rough_cut.mp4'
        self.log.info('Adding voiceover')
        mux_voiceover(assembled,Path(plan['audio']['source']),float(plan['target_duration']),self.cfg,final)
        self.log.info('Running final QC')
        qc=validate_output(final,plan,self.cfg); qc['preflight']=pf; write_json(self.ws/'qc_report.json',qc)
        summary={"clips_total":len(clip_items),"clips_rendered":clips_rendered,"clips_cached":clips_cached,"gap_fillers":gap_count,"duration":qc.get('duration'),"encoder":encoder,"qc_passed":qc['passed'],"output":str(final.resolve())}
        write_json(self.ws/'run_summary.json',summary)
        if not qc['passed']:
            raise RuntimeError('Agent 3 QC failed: '+'; '.join(qc['issues']))
        return summary
