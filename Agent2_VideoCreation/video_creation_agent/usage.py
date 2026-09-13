from __future__ import annotations

class UsageLedger:
    def __init__(self,max_uses=2,min_gap=2):
        self.max=max_uses; self.gap=min_gap; self.data={}
    def info(self,cid):
        xs=self.data.get(cid,[])
        return {'count':len(xs),'last':max(xs) if xs else None,'segments':list(xs)}
    def can_use(self,cid,seg_idx,repeat=False):
        xs=self.data.get(cid,[])
        if len(xs)>=self.max:return False
        if not xs:return True
        if not repeat:return False
        # Never same/adjacent/too-close. Absolute gap also protects a global-review rematch
        # of an earlier segment when the clip is already used later in the timeline.
        return all(abs(seg_idx-x)>self.gap for x in xs)
    def mark(self,cid,seg_idx):
        self.data.setdefault(cid,[]).append(seg_idx)
    def used_clip_ids(self):
        return set(self.data)
    def used_entries(self):
        out=[]
        for cid,xs in self.data.items():
            for seg in xs: out.append((cid,seg))
        return sorted(out,key=lambda x:(x[1],x[0]))
    def release(self,cid,seg_idx=None):
        xs=self.data.get(cid,[])
        if not xs:return
        if seg_idx is None:
            xs.pop()
        else:
            try: xs.remove(seg_idx)
            except ValueError: return
        if xs:self.data[cid]=xs
        else:self.data.pop(cid,None)
