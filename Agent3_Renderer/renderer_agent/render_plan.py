from __future__ import annotations
from pathlib import Path
from .timeline_reader import flatten_clips

def _frame(t:float,fps:int)->int:
    return int(round(float(t)*fps))

def build_render_plan(timeline:dict,cfg,encoder:str)->dict:
    duration=float(timeline["audio"]["duration"]); total_frames=_frame(duration,cfg.fps)
    clips=flatten_clips(timeline)
    items=[]; cursor_frame=0
    for idx,c in enumerate(clips,1):
        start_frame=_frame(c["timeline_start"],cfg.fps); end_frame=_frame(c["timeline_end"],cfg.fps)
        start_frame=max(start_frame,cursor_frame)
        if start_frame>cursor_frame:
            items.append({"type":"gap","start_frame":cursor_frame,"end_frame":start_frame,"frame_count":start_frame-cursor_frame,"duration":(start_frame-cursor_frame)/cfg.fps})
        frame_count=max(0,end_frame-start_frame)
        if frame_count:
            items.append({"type":"clip","ordinal":idx,"clip_id":c.get("clip_id",f"clip_{idx:04d}"),"segment_id":c.get("segment_id"),"source":c["source"],"source_start":float(c["source_start"]),"source_end":float(c["source_end"]),"timeline_start":start_frame/cfg.fps,"timeline_end":end_frame/cfg.fps,"start_frame":start_frame,"end_frame":end_frame,"frame_count":frame_count,"duration":frame_count/cfg.fps})
        cursor_frame=max(cursor_frame,end_frame)
    if cursor_frame<total_frames:
        items.append({"type":"gap","start_frame":cursor_frame,"end_frame":total_frames,"frame_count":total_frames-cursor_frame,"duration":(total_frames-cursor_frame)/cfg.fps})
    if cursor_frame>total_frames and items:
        # Clip the last timeline item to the audio target frame count.
        over=cursor_frame-total_frames
        last=items[-1]
        if last["frame_count"]>over:
            last["frame_count"]-=over; last["end_frame"]-=over; last["duration"]=last["frame_count"]/cfg.fps
            if last["type"]=="clip": last["timeline_end"]=last["end_frame"]/cfg.fps
    return {"schema_version":1,"target_duration":duration,"target_frames":total_frames,"output":{"width":cfg.width,"height":cfg.height,"fps":cfg.fps,"encoder":encoder,"pixel_format":cfg.pixel_format},"audio":{"source":timeline["audio"]["source"]},"items":items}
