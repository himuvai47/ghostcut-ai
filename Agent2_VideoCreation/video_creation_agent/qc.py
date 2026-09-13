from __future__ import annotations

def validate_timeline(timeline,clip_map,max_uses=2,min_clip_seconds=None,max_clip_seconds=None):
    issues=[]; uses={}; uncertain_used=0; audio_d=float(timeline['audio']['duration'])
    last_end=0.0
    for s in timeline['segments']:
        a,b=float(s['timeline_start']),float(s['timeline_end'])
        if a< -1e-6 or b<=a or b>audio_d+0.05: issues.append(f"{s['segment_id']}: invalid timeline range {a}-{b}")
        if a < last_end-0.05: issues.append(f"{s['segment_id']}: timeline overlap")
        last_end=max(last_end,b)
        if s['status'] not in {'matched','no_match'}: issues.append(f"{s['segment_id']}: invalid status")
        if s.get('status')=='matched' and s.get('clips') and all('timeline_start' in c and 'timeline_end' in c for c in s.get('clips',[])):
            spans=sorted((float(c['timeline_start']),float(c['timeline_end'])) for c in s.get('clips',[]))
            cursor=a; uncovered=0.0
            for x,y in spans:
                if x>cursor+0.05: uncovered+=x-cursor
                cursor=max(cursor,y)
            if cursor<b-0.05: uncovered+=b-cursor
            if uncovered>0.10:
                issues.append(f"{s['segment_id']}: matched segment leaves {uncovered:.2f}s uncovered")
        for c in s.get('clips',[]):
            cid=c['clip_id']; uses[cid]=uses.get(cid,0)+1
            src=clip_map.get(cid)
            if not src: issues.append(f"{s['segment_id']}: unknown clip {cid}"); continue
            if not src.usable: issues.append(f"{s['segment_id']}: unusable clip selected {cid}")
            face_status=c.get('face_status')
            if face_status=='face_visible' or face_status not in {'no_face','uncertain'}: issues.append(f"{s['segment_id']}: face gate not passed for {cid} ({face_status})")
            if face_status=='uncertain': uncertain_used+=1
            if c['source_start'] < src.start_time-0.02 or c['source_end']>src.end_time+0.02 or c['source_end']<=c['source_start']: issues.append(f"{s['segment_id']}: source range outside semantic clip {cid}")
            cut_d=float(c.get('timeline_end',0))-float(c.get('timeline_start',0))
            if min_clip_seconds is not None and cut_d < float(min_clip_seconds)-0.05: issues.append(f"{s['segment_id']}: cut {cid} is {cut_d:.2f}s (min {float(min_clip_seconds):.2f}s)")
            if max_clip_seconds is not None and cut_d > float(max_clip_seconds)+0.05: issues.append(f"{s['segment_id']}: cut {cid} is {cut_d:.2f}s (max {float(max_clip_seconds):.2f}s)")
    for cid,n in uses.items():
        if n>max_uses: issues.append(f"{cid}: used {n} times (max {max_uses})")
    return {'passed':not issues,'issues':issues,'clip_use_counts':uses,'uncertain_face_clips_used':uncertain_used,'segment_count':len(timeline['segments']),'matched_segments':sum(1 for s in timeline['segments'] if s['status']=='matched'),'no_match_segments':sum(1 for s in timeline['segments'] if s['status']=='no_match')}
