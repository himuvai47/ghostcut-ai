from __future__ import annotations
import math,re
from .models import IndexedClip,Candidate,VisualRequirement

def clip_text(c:IndexedClip)->str:
    parts=[c.description,c.primary_subject,c.primary_product,*c.secondary_products,*c.objects,*c.actions,c.environment,c.shot_type,c.content_type,*c.visual_attributes,*c.visual_concepts,*c.visible_text,*c.observed_details]
    # Agent 1 v1.3 metadata is compact, visually grounded retrieval context. Legacy
    # clips omit it entirely so partial backfills remain migration-safe.
    if getattr(c,'has_editorial_profile',False):
        parts.extend([
            c.brand,c.model_name,c.subject_focus,c.shot_angle,c.editorial_role,
            c.visual_energy,c.lighting_style,c.composition,c.visual_family,
        ])
    return '\n'.join(str(x) for x in parts if x and x not in {'none','unknown','other'})
def req_text(r:VisualRequirement)->str:
    return '\n'.join([r.requested_visual,r.primary_subject,*r.required_entities,*r.required_attributes,*r.preferred_shot_types,*r.preferred_content_types,*r.visual_concepts,*r.avoid])
def cosine(a,b):
    d=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b)); return d/(na*nb) if na and nb else 0.0
def toks(s): return set(re.findall(r'[a-z0-9]+',s.casefold()))
def lexical(a,b):
    x,y=toks(a),toks(b); return len(x&y)/max(1,len(x|y))
def structured(r,c):
    score=.5
    if r.preferred_shot_types and c.shot_type in r.preferred_shot_types: score+=.18
    if r.preferred_content_types and c.content_type in r.preferred_content_types: score+=.18
    low=' '.join([c.primary_product,c.primary_subject,c.description,*c.visible_text]).casefold()
    if r.required_entities:
        hits=sum(1 for e in r.required_entities if e.casefold() in low); score += .14*(hits/max(1,len(r.required_entities)))
    return min(1.0,score)
def rank(r,clips,req_emb,clip_embs,usage):
    out=[]
    for c in clips:
        if not c.usable: continue
        sem=max(0.0,cosine(req_emb,clip_embs[c.clip_id])); lex=lexical(req_text(r),clip_text(c)); st=structured(r,c); a1=max(0,min(1,c.analysis_confidence))
        total=.56*sem+.16*lex+.18*st+.10*a1
        u=usage.get(c.clip_id,{'count':0,'last':None})
        out.append(Candidate(c,sem,lex,st,a1,total,usage_count=u['count'],last_used_segment=u['last']))
    return sorted(out,key=lambda x:x.retrieval_score,reverse=True)
