from __future__ import annotations

def choose_window(clip,needed,use_number):
    take=min(needed,clip.duration)
    if clip.duration<=take+0.05:return clip.start_time,clip.start_time+take
    spare=clip.duration-take
    frac=.5 if use_number==1 else .8
    start=clip.start_time+spare*frac
    return start,start+take
