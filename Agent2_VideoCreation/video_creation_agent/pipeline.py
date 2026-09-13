from __future__ import annotations
import json, math, re
from dataclasses import replace
from pathlib import Path
from .agent1_reader import load_agent1_clips
from .audio_alignment import build_alignment
from .cache import RetrievalCache
from .config import Agent2Config
from .global_review import review as global_review
from .models import VisualRequirement, TimelineClip, CandidateJudgment
from .ollama import embed
from .planner import plan_segments, PLANNER_VERSION
from .qc import validate_timeline
from .rerank import rerank, RERANK_VERSION
from .retrieval import clip_text, req_text, rank, cosine
from .source_window import choose_window
from .usage import UsageLedger
from .utils import write_json, read_json, sha256_file, sha256_text

LEVEL_ORDER={'strong_match':0,'acceptable_match':1,'weak_match':2,'reject':3}
LEVEL_BASE={'strong_match':.92,'acceptable_match':.78,'weak_match':.58,'reject':.0}
MATCH_POLICY_VERSION='match-v13-editorial-vision-adapter'
GAP_FILL_MAX_CANDIDATES=2
GAP_FILL_SCAN_LIMIT=64
EDITORIAL_PROFILE_MIN_VERSION=1
EDITORIAL_VISUAL_FAMILY_RECENT_PENALTY=0.34
EDITORIAL_VISUAL_FAMILY_REUSE_PENALTY=0.18
EDITORIAL_ROLE_FOCUS_PREVIOUS_PENALTY=0.08
EDITORIAL_ROLE_FOCUS_RECENT_PENALTY=0.04

def unresolved_major_issues(review_issues):
    return [x for x in review_issues if x.get('severity')=='major' and x.get('action')!='keep' and not x.get('resolved',False)]


def review_risks(timeline_segments):
    """Return only conditions worth paying for a second Qwen editorial pass.

    Fine-detail coverage is intentionally not a risk in faceless B-roll. The normal
    reranker already checks subject/product relevance. Global review is reserved
    for weak selections or deliberate clip reuse, where a second opinion can still
    change the edit materially.
    """
    risks=[]
    for seg in timeline_segments:
        for c in seg.get('clips',[]):
            # A weak-but-related B-roll choice is acceptable in faceless editing.
            # Only deliberate reuse is worth paying for a second editorial pass.
            if int(c.get('use_number') or 1)>1: risks.append(f"{seg.get('segment_id')}:repeat")
    return list(dict.fromkeys(risks))

class Agent2Pipeline:
    def __init__(self,cfg:Agent2Config,workspace:Path,root:Path,logger):
        self.cfg=cfg; self.ws=workspace; self.root=root; self.log=logger
        self.cache=RetrievalCache(workspace/'cache'/'retrieval_cache.db')
    def close(self): self.cache.close()

    def _editorial_requirement(self, req):
        """Return the effective retrieval requirement for this editorial mode.

        Literal beats retain the planner's exact requirement. Thematic and filler
        beats deliberately drop sentence-level hard detail so ordinary same-topic
        footage can win on visual quality/diversity instead of fake literalness.
        """
        if not bool(getattr(self.cfg,'editorial_freedom_enabled',True)):
            return req
        mode=str(getattr(req,'match_mode','thematic') or 'thematic')
        if mode=='literal':
            return req
        anchor=' '.join(str(getattr(req,'context_anchor','') or '').split())
        subject=anchor or ' '.join(str(req.primary_subject or '').split())
        if mode=='brand_filler':
            requested=(f'Attractive varied real B-roll of {subject}. General same-brand or same-topic footage is intentional editorial filler; exact narration details do not need to be visible.'
                       if subject else
                       'Attractive varied real topical B-roll. Exact narration details do not need to be visible.')
            return replace(req,
                requested_visual=requested,
                primary_subject=subject,
                required_entities=[],required_attributes=[],
                preferred_shot_types=[],preferred_content_types=[],
                visual_concepts=[subject] if subject else [],
                specificity='conceptual',visual_intent='conceptual')
        requested=(f'Relevant real B-roll of {subject} supporting the topic. Broad topical relevance is enough; abstract narration details do not need literal depiction.'
                   if subject else
                   'Relevant real topical B-roll supporting the narration. Broad topical relevance is enough.')
        return replace(req,
            requested_visual=requested,
            primary_subject=subject,
            required_entities=[],required_attributes=[],
            specificity='normal')

    def _clip_matches_anchor(self, anchor, clip):
        anchor=' '.join(str(anchor or '').casefold().split())
        if not anchor:
            return False
        text=' '.join([
            str(clip.primary_product or ''),str(clip.primary_subject or ''),str(clip.description or ''),
            str(getattr(clip,'brand','') or ''),str(getattr(clip,'model_name','') or ''),
            str(getattr(clip,'subject_focus','') or ''),str(getattr(clip,'editorial_role','') or ''),
            *[str(x) for x in (clip.secondary_products or [])],
            *[str(x) for x in (clip.objects or [])],
            *[str(x) for x in (clip.visible_text or [])],
        ]).casefold()
        if anchor in text:
            return True
        stop={'watch','watches','luxury','product','products','brand','brands','real','b-roll','broll'}
        toks=[x for x in re.findall(r'[a-z0-9]+',anchor) if x not in stop]
        hay=set(re.findall(r'[a-z0-9]+',text))
        return bool(toks) and all(x in hay for x in toks)

    def _pick_brand_filler(self,req,seg_idx,clips,clip_embs,ledger,retrieval_log,req_emb=None,exclude_ids=None):
        """Choose strong same-context filler without a Qwen rerank call.

        This is intentional editorial freedom, not a matching failure. Exact anchor
        hits are preferred, but diverse related unused footage can fill when the
        library does not contain the named brand/model.
        """
        if req_emb is None:
            req_emb=embed(self.cfg.ollama_url,self.cfg.embedding_model,[req_text(req)])[0]
        usage={c.clip_id:ledger.info(c.clip_id) for c in clips}
        ranked=rank(req,clips,req_emb,clip_embs,usage)
        ranked=self._apply_visual_diversity(ranked,clips,clip_embs,ledger,seg_idx,req)
        excluded=set(exclude_ids or [])
        if excluded:
            ranked=[c for c in ranked if c.clip.clip_id not in excluded]
        face_free,checked=self._face_free_shortlist(ranked)
        retrieval_log.append({
            'segment_id':req.segment_id,'requested_visual':req.requested_visual,
            'match_mode':'brand_filler','context_anchor':getattr(req,'context_anchor',''),
            'fallback_mode':'intentional_brand_filler',
            'candidates':[dict(c.compact_dict(),face_status=c.face_status,face_reason=c.face_reason) for c in checked],
        })
        if not face_free:
            return [],{'segment_id':req.segment_id,'status':'no_match','reason':'No allowed candidate survived Agent 1 face metadata filtering for editorial filler.','requested_visual':req.requested_visual,'match_mode':'brand_filler'}

        anchor=getattr(req,'context_anchor','') or req.primary_subject
        mn=float(self.cfg.min_visual_clip_seconds)
        anchor_floor=float(getattr(self.cfg,'brand_filler_min_retrieval_score',0.14))
        related_floor=float(self.cfg.broad_fallback_min_retrieval_score)
        selected=[]; remain=req.duration; used_here=set()

        def ordered_pool(repeat=False):
            legal=[]
            for c in face_free:
                info=ledger.info(c.clip.clip_id)
                if c.clip.duration < mn-0.03: continue
                if repeat:
                    if info['count']!=1 or not ledger.can_use(c.clip.clip_id,seg_idx,repeat=True): continue
                else:
                    if info['count']!=0 or not ledger.can_use(c.clip.clip_id,seg_idx,repeat=False): continue
                hit=self._clip_matches_anchor(anchor,c.clip)
                floor=anchor_floor if hit else related_floor
                if c.retrieval_score < floor: continue
                legal.append((c,hit))
            # Context first, but a visually fresh topical shot beats a near-duplicate
            # context shot. This preserves v1.0.14's diversity behavior.
            groups=[]
            for hit,near in ((True,False),(False,False),(True,True),(False,True)):
                group=[c for c,h in legal if h==hit and bool(c.near_duplicate)==near]
                if self.cfg.prefer_longer_clips:
                    group=sorted(group,key=lambda c:(-min(float(c.clip.duration),float(self.cfg.max_visual_clip_seconds)),-self._adjusted_candidate_score(c)))
                groups.extend(group)
            return groups

        def consume(pool,repeat=False):
            nonlocal remain
            for cand in pool:
                if remain<=0.03: return
                cid=cand.clip.clip_id
                if cid in used_here: continue
                take=self._planned_take(remain,cand.clip.duration)
                if take<=0.03: continue
                use_no=ledger.info(cid)['count']+1
                ss,se=choose_window(cand.clip,take,use_no)
                hit=self._clip_matches_anchor(anchor,cand.clip)
                reason=('Same-brand/context B-roll selected as intentional editorial filler.' if hit else
                        'Related topical B-roll selected as intentional editorial filler.')
                j=CandidateJudgment(cid,'strong' if hit else 'partial','partial','partial','partial',False,[],
                                    'acceptable_match',reason)
                conf=max(0.52,min(0.86,float(cand.retrieval_score)+(.20 if hit else .12)))
                selected.append({'clip':cand.clip,'judgment':j,'source_start':ss,'source_end':se,'take':take,
                                 'confidence':round(conf,3),'use_number':use_no,'face_status':cand.face_status,
                                 'editorial_filler':True})
                ledger.mark(cid,seg_idx); used_here.add(cid); remain-=take

        consume(ordered_pool(False),False)
        if remain>=mn-0.03 and self._unused_legal_count(clips,ledger) < int(self.cfg.min_unused_clips_before_repeat):
            consume(ordered_pool(True),True)
        if not selected:
            return [],{'segment_id':req.segment_id,'status':'no_match','reason':'No unused face-safe same-context or related filler footage was available.','requested_visual':req.requested_visual,'match_mode':'brand_filler'}
        dec={'segment_id':req.segment_id,'status':'matched','requested_visual':req.requested_visual,
             'match_mode':'brand_filler','context_anchor':anchor,'coverage_shortfall':round(max(0,remain),3),
             'selected':[{'clip_id':x['clip'].clip_id,'decision':x['judgment'].decision,'reason':x['judgment'].reason,
                          'confidence':x['confidence'],'use_number':x['use_number'],'editorial_filler':True} for x in selected]}
        if remain>0.03: dec['warning']=f'Coverage shortfall {remain:.2f}s after exhausting editorial filler candidates.'
        return selected,dec

    def _unused_legal_count(self,clips,ledger):
        info_fn=getattr(ledger,'info',None)
        allowed={'no_face','uncertain'} if self.cfg.allow_uncertain_face_status else {'no_face'}
        total=0
        for c in clips:
            info=info_fn(c.clip_id) if info_fn else {'count':0}
            if int(info.get('count',0))==0 and c.usable and c.face_status in allowed and c.duration>=self.cfg.min_visual_clip_seconds-0.03:
                total+=1
        return total

    def _has_editorial_profile(self, clip):
        return int(getattr(clip,'editorial_profile_version',0) or 0) >= EDITORIAL_PROFILE_MIN_VERSION

    @staticmethod
    def _editorial_value(value):
        text=' '.join(str(value or '').split()).strip()
        return '' if text.casefold() in {'','unknown','none','other','n/a','null'} else text

    def _editorial_metadata_bonus(self, cand, req=None):
        """Small positive tie-breaker from Agent 1 v1.3 Editorial Vision metadata.

        Legacy clips receive exactly zero bonus and no penalty. Relevance floors are
        still evaluated on the original retrieval score, so metadata cannot make an
        irrelevant clip pass matching gates.
        """
        c=cand.clip
        cand.editorial_bonus=0.0; cand.editorial_reasons=[]
        if not self._has_editorial_profile(c):
            return 0.0
        bonus=0.0; reasons=[]
        try:
            quality=max(0.0,min(1.0,float(getattr(c,'visual_quality_score',0.5))))
            usefulness=max(0.0,min(1.0,float(getattr(c,'editorial_usefulness_score',0.5))))
        except (TypeError,ValueError):
            quality=usefulness=0.5
        strength=(quality+usefulness)/2.0
        if strength>0.65:
            q=min(0.04,(strength-0.65)*0.12)
            bonus+=q; reasons.append(f'editorial_quality:{strength:.2f}')

        if req is not None:
            mode=str(getattr(req,'match_mode','thematic') or 'thematic')
            anchor=self._editorial_value(getattr(req,'context_anchor','') or getattr(req,'primary_subject',''))
            metadata=' '.join(filter(None,[self._editorial_value(getattr(c,'brand','')),self._editorial_value(getattr(c,'model_name',''))])).casefold()
            if anchor and metadata:
                stop={'watch','watches','luxury','product','products','brand','brands','real','b-roll','broll','vs','and'}
                at=[x for x in re.findall(r'[a-z0-9]+',anchor.casefold()) if x not in stop]
                hay=set(re.findall(r'[a-z0-9]+',metadata))
                if at and all(x in hay for x in at):
                    bonus+=0.035; reasons.append('editorial_brand_model_anchor')

            role=self._editorial_value(getattr(c,'editorial_role','')).casefold()
            focus=self._editorial_value(getattr(c,'subject_focus','')).casefold()
            requested=' '.join([str(getattr(req,'requested_visual','') or ''),str(getattr(req,'narration_text','') or ''),str(getattr(req,'primary_subject','') or '')]).casefold()
            filler_roles={'product_beauty','technical_detail','movement_detail','wrist_lifestyle','handling_demo','retail_display','brand_identity','comparison_support','generic_topic_broll'}
            if mode=='brand_filler' and role in filler_roles:
                bonus+=0.025; reasons.append(f'filler_role:{role}')
            elif mode=='thematic' and role in filler_roles:
                bonus+=0.015; reasons.append(f'thematic_role:{role}')

            if re.search(r'\b(movement|caliber|calibre|caseback|mechanical|mechanism|eta|sellita|duw)\b',requested):
                if role in {'movement_detail','technical_detail'} or focus in {'movement','caseback'}:
                    bonus+=0.04; reasons.append('literal_technical_focus')
            elif re.search(r'\b(dial|bezel|bracelet|strap|crown|pusher|case)\b',requested):
                if focus and focus in requested.replace('-','_').replace(' ','_'):
                    bonus+=0.03; reasons.append(f'literal_focus:{focus}')
            elif re.search(r'\b(logo|brand|branding)\b',requested):
                if role=='brand_identity' or focus=='brand_logo':
                    bonus+=0.03; reasons.append('brand_identity_focus')

        cand.editorial_bonus=min(0.10,bonus)
        cand.editorial_reasons=list(dict.fromkeys(reasons))
        return cand.editorial_bonus

    @staticmethod
    def _adjusted_candidate_score(cand):
        return float(cand.retrieval_score)-float(getattr(cand,'diversity_penalty',0.0))+float(getattr(cand,'editorial_bonus',0.0))

    def _apply_visual_diversity(self, ranked, clips, clip_embs, ledger, seg_idx, req=None):
        """Re-order candidates to avoid footage that only looks unique by clip ID.

        Uses only persistent Agent-1 metadata and the embeddings already cached for
        retrieval. No new model or image call is made. Relevance remains primary,
        while same-source recency, nearby source-time windows, source imbalance and
        near-identical semantic/framing matches receive deterministic penalties.
        """
        if not ranked:
            return ranked
        clip_by_id={c.clip_id:c for c in clips}
        used=[]
        source_counts={}
        for cid,used_seg in ledger.used_entries():
            uc=clip_by_id.get(cid)
            if not uc: continue
            used.append((uc,used_seg))
            source_counts[uc.source_id]=source_counts.get(uc.source_id,0)+1
        all_sources={c.source_id for c in clips if c.usable}
        for sid in all_sources: source_counts.setdefault(sid,0)
        min_source=min(source_counts.values()) if source_counts else 0
        recent=[(c,si) for c,si in used if 0 < seg_idx-si <= max(4,int(self.cfg.diversity_source_cooldown_segments)+2)]
        profiled_used=[(c,si) for c,si in used if self._has_editorial_profile(c)]
        profiled_recent=[(c,si) for c,si in recent if self._has_editorial_profile(c)]
        threshold=float(self.cfg.diversity_semantic_duplicate_threshold)
        near_secs=float(self.cfg.diversity_same_source_near_seconds)
        cooldown=max(0,int(self.cfg.diversity_source_cooldown_segments))
        for cand in ranked:
            c=cand.clip; penalty=0.0; reasons=[]; near_dup=False
            self._editorial_metadata_bonus(cand,req)
            # Agent 1 v1.3 visual families are a stronger signal than clip IDs or
            # description similarity. An exact family reuse is visually repetitive even
            # when it comes from a different source file or timestamp.
            if self._has_editorial_profile(c):
                family=self._editorial_value(getattr(c,'visual_family','')).casefold()
                if family:
                    recent_family=[seg_idx-si for uc,si in profiled_recent if self._editorial_value(getattr(uc,'visual_family','')).casefold()==family]
                    any_family=any(self._editorial_value(getattr(uc,'visual_family','')).casefold()==family for uc,_ in profiled_used)
                    if recent_family:
                        penalty+=EDITORIAL_VISUAL_FAMILY_RECENT_PENALTY
                        reasons.append('agent1_visual_family_recent')
                        near_dup=True
                    elif any_family:
                        penalty+=EDITORIAL_VISUAL_FAMILY_REUSE_PENALTY
                        reasons.append('agent1_visual_family_reuse')

                role=self._editorial_value(getattr(c,'editorial_role','')).casefold()
                focus=self._editorial_value(getattr(c,'subject_focus','')).casefold()
                if role and focus and profiled_recent:
                    closest=sorted(profiled_recent,key=lambda x:seg_idx-x[1])[0][0]
                    cr=self._editorial_value(getattr(closest,'editorial_role','')).casefold()
                    cf=self._editorial_value(getattr(closest,'subject_focus','')).casefold()
                    if role==cr and focus==cf:
                        gap=seg_idx-max(si for uc,si in profiled_recent if uc.clip_id==closest.clip_id)
                        x=EDITORIAL_ROLE_FOCUS_PREVIOUS_PENALTY if gap<=1 else EDITORIAL_ROLE_FOCUS_RECENT_PENALTY
                        penalty+=x; reasons.append('agent1_same_role_focus_recent')
                    elif role==cr or focus==cf:
                        penalty+=0.02; reasons.append('agent1_similar_role_or_focus')
            # Source balancing: if several videos can satisfy the topic, do not let one
            # source dominate merely because its descriptions score a few points higher.
            excess=max(0,source_counts.get(c.source_id,0)-min_source)
            if excess:
                x=min(0.12,float(self.cfg.diversity_source_balance_penalty)*excess)
                penalty+=x; reasons.append(f'source_balance:{excess}')
            # Rotate away from the source video used in the immediately preceding beats.
            same_recent=[seg_idx-si for uc,si in recent if uc.source_id==c.source_id]
            if same_recent:
                gap=min(same_recent)
                if gap<=1: penalty+=0.15; reasons.append('same_source_previous_beat')
                elif gap<=cooldown: penalty+=0.09; reasons.append(f'same_source_within_{cooldown}_beats')
                else: penalty+=0.035; reasons.append('same_source_recent')
            # A different Agent-1 clip from the same few seconds of a source video is
            # visually a repeat even when the clip IDs differ.
            for uc,_ in used:
                if uc.source_id!=c.source_id: continue
                gap=max(0.0,max(float(uc.start_time)-float(c.end_time),float(c.start_time)-float(uc.end_time)))
                if gap<=near_secs:
                    penalty+=0.32 if gap<=5.0 else 0.22
                    reasons.append(f'nearby_source_time:{gap:.1f}s'); near_dup=True; break
            # Description embeddings catch duplicate-looking shots across separate
            # source files. Require similar framing/content before treating them as a
            # near duplicate so all TAG-Heuer close-ups are not blindly collapsed.
            cv=clip_embs.get(c.clip_id)
            if cv is not None:
                best_sim=0.0; best_uc=None
                for uc,_ in recent:
                    uv=clip_embs.get(uc.clip_id)
                    if uv is None: continue
                    sim=cosine(cv,uv)
                    if sim>best_sim: best_sim=sim; best_uc=uc
                if best_uc is not None:
                    same_family=(c.shot_type==best_uc.shot_type or c.content_type==best_uc.content_type)
                    if same_family and best_sim>=threshold:
                        penalty+=0.26; reasons.append(f'near_duplicate_embedding:{best_sim:.3f}'); near_dup=True
                    elif same_family and best_sim>=max(0.90,threshold-0.04):
                        penalty+=0.08; reasons.append(f'similar_visual_family:{best_sim:.3f}')
            cand.diversity_penalty=min(0.65,penalty)
            cand.diversity_reasons=list(dict.fromkeys(reasons))
            cand.near_duplicate=near_dup
        # For ordinary faceless B-roll, hard-deprioritize near duplicates when a healthy
        # alternative pool exists. Strict/high named-product beats keep them available.
        strict=bool(req is not None and req.specificity in {'strict','high'})
        diverse=[c for c in ranked if not c.near_duplicate]
        if not strict and len(diverse)>=int(self.cfg.diversity_min_alternatives):
            ranked=diverse+[c for c in ranked if c.near_duplicate]
        def adjusted(c): return self._adjusted_candidate_score(c)
        # Preserve the diverse-first partition while sorting each partition by adjusted relevance.
        if not strict and len(diverse)>=int(self.cfg.diversity_min_alternatives):
            a=sorted(diverse,key=adjusted,reverse=True)
            b=sorted([c for c in ranked if c.near_duplicate],key=adjusted,reverse=True)
            return a+b
        return sorted(ranked,key=adjusted,reverse=True)

    def _diversity_metrics(self,timeline_segments,clip_map,clip_embs):
        ordered=[]
        for seg in timeline_segments:
            for tc in seg.get('clips',[]):
                c=clip_map.get(tc.get('clip_id'))
                if c: ordered.append(c)
        distribution={}
        same_source_consecutive=0; nearby_source_reuses=0; semantic_near_duplicates=0
        visual_family_reuses=0; same_role_focus_consecutive=0; profiled_used_ids=set()
        unique_families=set(); seen_families=set()
        seen=[]; threshold=float(self.cfg.diversity_semantic_duplicate_threshold)
        near_secs=float(self.cfg.diversity_same_source_near_seconds)
        for i,c in enumerate(ordered):
            distribution[c.source_video]=distribution.get(c.source_video,0)+1
            if i and ordered[i-1].source_id==c.source_id: same_source_consecutive+=1
            if self._has_editorial_profile(c):
                profiled_used_ids.add(c.clip_id)
                fam=self._editorial_value(getattr(c,'visual_family','')).casefold()
                if fam:
                    if fam in seen_families: visual_family_reuses+=1
                    seen_families.add(fam); unique_families.add(fam)
                if i and self._has_editorial_profile(ordered[i-1]):
                    prev=ordered[i-1]
                    role=self._editorial_value(getattr(c,'editorial_role','')).casefold()
                    focus=self._editorial_value(getattr(c,'subject_focus','')).casefold()
                    prole=self._editorial_value(getattr(prev,'editorial_role','')).casefold()
                    pfocus=self._editorial_value(getattr(prev,'subject_focus','')).casefold()
                    if role and focus and role==prole and focus==pfocus:
                        same_role_focus_consecutive+=1
            nearby=False; semdup=False
            for prev in seen:
                if prev.source_id==c.source_id:
                    gap=max(0.0,max(float(prev.start_time)-float(c.end_time),float(c.start_time)-float(prev.end_time)))
                    if gap<=near_secs: nearby=True
                a=clip_embs.get(c.clip_id); b=clip_embs.get(prev.clip_id)
                if a is not None and b is not None and (c.shot_type==prev.shot_type or c.content_type==prev.content_type):
                    if cosine(a,b)>=threshold: semdup=True
            nearby_source_reuses+=int(nearby); semantic_near_duplicates+=int(semdup)
            seen.append(c)
        n=max(1,len(ordered)-1)
        duplicate_events=max(nearby_source_reuses,semantic_near_duplicates,visual_family_reuses)
        score=max(0.0,min(1.0,1.0-(duplicate_events+0.5*same_source_consecutive+0.35*same_role_focus_consecutive)/n))
        return {
            'source_distribution':distribution,
            'same_source_consecutive_cuts':same_source_consecutive,
            'nearby_source_region_reuses':nearby_source_reuses,
            'semantic_near_duplicate_reuses':semantic_near_duplicates,
            'visual_family_reuses':visual_family_reuses,
            'same_editorial_role_focus_consecutive':same_role_focus_consecutive,
            'agent1_editorial_profile_clips_used':len(profiled_used_ids),
            'unique_visual_families_used':len(unique_families),
            'visual_diversity_score':round(score,3),
        }

    def _clip_embedding_cache_hash(self, clip):
        # Legacy clips keep the exact v1.0.19 cache key. Profiled clips use a new
        # namespace because clip_text() now includes Agent 1 v1.3 editorial fields.
        # This refreshes only enhanced clips after a partial backfill, not all 570.
        if self._has_editorial_profile(clip):
            return sha256_text('clip-text-v2-agent1-editorial-vision\n'+str(clip.analysis_hash))
        return clip.analysis_hash

    def _ensure_embeddings(self,clips):
        missing=[]; vecs={}
        for c in clips:
            cache_hash=self._clip_embedding_cache_hash(c)
            v=self.cache.get_embedding(c.clip_id,cache_hash,self.cfg.embedding_model)
            if v is None: missing.append(c)
            else: vecs[c.clip_id]=v
        self.log.info(f"Retrieval embeddings: {len(vecs)} cached / {len(missing)} missing")
        for i in range(0,len(missing),32):
            batch=missing[i:i+32]; embs=embed(self.cfg.ollama_url,self.cfg.embedding_model,[clip_text(c) for c in batch])
            if len(embs)!=len(batch): raise RuntimeError('Embedding result count mismatch')
            for c,v in zip(batch,embs):
                self.cache.put_embedding(c.clip_id,self._clip_embedding_cache_hash(c),self.cfg.embedding_model,v); vecs[c.clip_id]=v
        return vecs

    def _face_free_shortlist(self, ranked):
        accepted=[]; checked=[]
        # v1.0.5: face status is persistent Agent-1 metadata. Agent 2 performs zero
        # image extraction and zero Qwen2.5-VL calls for face screening.
        for cand in ranked[:self.cfg.retrieval_top_k]:
            status=cand.clip.face_status
            cand.face_status=status
            cand.face_reason=f"Agent 1 persistent face metadata v{cand.clip.face_scan_version}"
            checked.append(cand)
            if status=='no_face' or (status=='uncertain' and self.cfg.allow_uncertain_face_status):
                accepted.append(cand)
        return accepted,checked

    def _candidate_order(self,judgments,candidates):
        jby={j.clip_id:j for j in judgments}
        pairs=[(c,jby.get(c.clip.clip_id)) for c in candidates if jby.get(c.clip.clip_id)]
        def face_adjusted(c):
            penalty=0.04 if c.face_status=='uncertain' else 0.0
            return c.retrieval_score-penalty-float(getattr(c,'diversity_penalty',0.0))+float(getattr(c,'editorial_bonus',0.0))
        def usable_len(c):
            return min(float(c.clip.duration),float(self.cfg.max_visual_clip_seconds))
        # Match level remains primary. Inside the same quality level, prefer a clip
        # that can stay on screen longer, then use retrieval relevance as tie-breaker.
        if self.cfg.prefer_longer_clips:
            return sorted(pairs,key=lambda x:(LEVEL_ORDER[x[1].decision],-usable_len(x[0]),-face_adjusted(x[0])))
        return sorted(pairs,key=lambda x:(LEVEL_ORDER[x[1].decision],-face_adjusted(x[0]),-usable_len(x[0])))

    def _planned_take(self,remain,clip_duration):
        """Return a legal 3-8 second cut that avoids leaving a sub-3s flash remainder."""
        remain=max(0.0,float(remain)); clip_duration=max(0.0,float(clip_duration))
        mn=float(self.cfg.min_visual_clip_seconds); mx=float(self.cfg.max_visual_clip_seconds)
        if remain<=0.03: return 0.0
        if remain<mn-0.03: return 0.0
        cap=min(remain,clip_duration,mx)
        if cap<mn-0.03: return 0.0
        rest=remain-cap
        if 0.03 < rest < mn-0.03:
            adjusted=remain-mn
            if adjusted>=mn-0.03 and adjusted<=clip_duration+0.03 and adjusted<=mx+0.03:
                cap=adjusted
            elif cap < remain-0.03:
                return 0.0
        return max(0.0,min(cap,mx))

    def _coverage_possible(self,pairs,target):
        remain=max(0.0,float(target))
        for cand,j in pairs:
            if j.hard_requirement_failed or j.decision=='reject': continue
            take=self._planned_take(remain,cand.clip.duration)
            if take<=0: continue
            remain-=take
            if remain<=0.03: return True
        return remain<=0.03

    def _judge_pool(self, req, pool, target_duration=None, allow_weak=True):
        pairs=[]
        # Progressive reranking: ask Qwen about the best four candidates first and
        # widen only when those do not provide enough acceptable footage. This keeps
        # the fallback depth of retrieval_top_k without paying for every candidate.
        batch_size=2
        target=max(0.03,float(target_duration if target_duration is not None else req.duration))
        for i in range(0,len(pool),batch_size):
            batch=pool[i:i+batch_size]
            js=rerank(self.cfg,req,batch)
            pairs.extend(self._candidate_order(js,batch))
            valid=[]
            for cand,j in pairs:
                if j.hard_requirement_failed or j.decision=='reject': continue
                if j.decision=='weak_match' and (not allow_weak or req.specificity in {'strict','high'}): continue
                valid.append((cand,j))
            # Stop once the judged pool can cover the unit using only legal 3-8s cuts.
            ordered=self._candidate_order([j for _,j in valid],[c for c,_ in valid]) if valid else []
            if ordered and self._coverage_possible(ordered,target):
                break
        return self._candidate_order([j for _,j in pairs],[c for c,_ in pairs])

    def _relaxed_gap_fill(self, req, seg_idx, clips, clip_embs, ledger, retrieval_log, remain, exclude_ids=None, repeat=False):
        """Fill a legal >=3s remainder with broadly related B-roll.

        The expensive/model-assisted portion stays tiny: Qwen judges at most two
        candidates. If those cannot legally complete the remaining duration, a
        deterministic coverage pass scans the already-ranked local candidates and
        selects unused topical B-roll that satisfies the 3-8s pacing rules. No
        additional embedding or Qwen call is required for that rescue path.
        """
        mn=float(self.cfg.min_visual_clip_seconds)
        if remain < mn-0.03:
            return []
        anchor=' '.join(str(getattr(req,'context_anchor','') or req.primary_subject or '').split())
        fallback_visual=(
            f'Attractive varied real B-roll of {anchor}. Exact narration details may be unavailable; same-brand or same-topic footage is appropriate.'
            if anchor else
            f'Broadly related topical B-roll supporting this narration: {req.narration_text}'
        )
        relaxed=replace(
            req,
            segment_id=f'{req.segment_id}__gap_fill',
            requested_visual=fallback_visual,
            primary_subject=anchor,
            required_entities=[],
            required_attributes=[],
            preferred_shot_types=[],
            preferred_content_types=[],
            specificity='conceptual',
            match_mode='brand_filler',
            timeline_start=0.0,
            timeline_end=remain,
        )
        req_emb=embed(self.cfg.ollama_url,self.cfg.embedding_model,[req_text(relaxed)])[0]
        usage={c.clip_id:ledger.info(c.clip_id) for c in clips}
        ranked=rank(relaxed,clips,req_emb,clip_embs,usage)
        ranked=self._apply_visual_diversity(ranked,clips,clip_embs,ledger,seg_idx,relaxed)
        excluded=set(exclude_ids or [])

        def legal_candidate(cand, score_floor):
            if cand.clip.clip_id in excluded:
                return False
            status=cand.clip.face_status
            cand.face_status=status
            cand.face_reason=f'Agent 1 persistent face metadata v{cand.clip.face_scan_version}'
            if status=='face_visible' or (status=='uncertain' and not self.cfg.allow_uncertain_face_status):
                return False
            if cand.clip.duration < mn-0.03 or cand.retrieval_score < score_floor:
                return False
            info=ledger.info(cand.clip.clip_id)
            if repeat:
                return info['count']==1 and ledger.can_use(cand.clip.clip_id,seg_idx,repeat=True)
            return info['count']==0 and ledger.can_use(cand.clip.clip_id,seg_idx,repeat=False)

        broad_floor=float(self.cfg.broad_fallback_min_retrieval_score)
        eligible=[]
        checked=[]
        for cand in ranked:
            if len(checked)<GAP_FILL_SCAN_LIMIT:
                checked.append(cand)
            if legal_candidate(cand,broad_floor):
                eligible.append(cand)
            if len(eligible)>=GAP_FILL_SCAN_LIMIT:
                break

        # If the normal broad threshold produced no usable duration shape, keep the
        # fallback moving by allowing the best semantically ranked legal library B-roll
        # at a lower emergency floor. This is still unused, face-safe, local footage.
        if not any(self._planned_take(remain,c.clip.duration)>0.03 for c in eligible):
            emergency_floor=max(0.0,min(broad_floor, broad_floor*0.5))
            seen={c.clip.clip_id for c in eligible}
            for cand in ranked:
                if cand.clip.clip_id in seen:
                    continue
                if legal_candidate(cand,emergency_floor):
                    eligible.append(cand); seen.add(cand.clip.clip_id)
                if len(eligible)>=GAP_FILL_SCAN_LIMIT:
                    break

        pool=eligible[:GAP_FILL_MAX_CANDIDATES]
        retrieval_log.append({
            'segment_id':relaxed.segment_id,
            'requested_visual':relaxed.requested_visual,
            'match_mode':'brand_filler','context_anchor':getattr(relaxed,'context_anchor',''),
            'fallback_mode':'general_gap_fill_repeat' if repeat else 'general_gap_fill_unused',
            'candidates':[dict(c.compact_dict(),face_status=c.face_status,face_reason=c.face_reason) for c in pool],
            'deterministic_scan_count':len(eligible),
        })
        if not eligible:
            return []

        out=[]
        left=float(remain)
        judged_ids=set()
        if pool:
            judgments=rerank(self.cfg,relaxed,pool)
            pairs=self._candidate_order(judgments,pool)
            for cand,j in pairs:
                judged_ids.add(cand.clip.clip_id)
                if left < mn-0.03:
                    break
                if j.hard_requirement_failed or j.decision=='reject':
                    continue
                take=self._planned_take(left,cand.clip.duration)
                if take<=0.03:
                    continue
                use_no=ledger.info(cand.clip.clip_id)['count']+1
                ss,se=choose_window(cand.clip,take,use_no)
                conf=max(0.0,min(1.0,.65*LEVEL_BASE[j.decision]+.35*cand.retrieval_score))
                out.append({
                    'clip':cand.clip,'judgment':j,'source_start':ss,'source_end':se,'take':take,
                    'confidence':round(conf,3),'use_number':use_no,'face_status':cand.face_status,
                    'gap_filler':True,
                })
                ledger.mark(cand.clip.clip_id,seg_idx)
                excluded.add(cand.clip.clip_id)
                left-=take

        # Coverage completion: if the two judged candidates were rejected or had the
        # wrong duration shape, walk the already-ranked unused pool locally. Prefer a
        # single clip that can finish the remainder; otherwise take the best legal cut
        # that leaves another >=3s remainder. This avoids the v1.0.12 4.47s gap bug.
        while left >= mn-0.03:
            viable=[]
            for cand in eligible:
                cid=cand.clip.clip_id
                if cid in excluded:
                    continue
                # Re-check because ledger may have changed after a judged selection.
                if not legal_candidate(cand,0.0):
                    continue
                take=self._planned_take(left,cand.clip.duration)
                if take>0.03:
                    viable.append((cand,take))
            if not viable:
                break
            full=[x for x in viable if abs(x[1]-left)<=0.03]
            cand,take=(full[0] if full else viable[0])
            use_no=ledger.info(cand.clip.clip_id)['count']+1
            ss,se=choose_window(cand.clip,take,use_no)
            j=CandidateJudgment(
                cand.clip.clip_id,'partial','partial','partial','partial',False,[],
                'acceptable_match',
                'Unused topic-related library B-roll selected deterministically to complete remaining timeline coverage.'
            )
            conf=max(0.52,min(0.78,float(cand.retrieval_score)+0.32))
            out.append({
                'clip':cand.clip,'judgment':j,'source_start':ss,'source_end':se,'take':take,
                'confidence':round(conf,3),'use_number':use_no,'face_status':cand.face_status,
                'gap_filler':True,
            })
            ledger.mark(cand.clip.clip_id,seg_idx)
            excluded.add(cand.clip.clip_id)
            left-=take
        return out

    def _collective_entities(self, req):
        # required_entities are the planner's must-show named entities. If more than
        # one survives normalization, satisfy them across sequential clips rather than
        # demanding one impossible source clip contain all of them.
        out=[]
        for e in req.required_entities:
            e=' '.join(str(e or '').split())
            if e and e.casefold() not in {x.casefold() for x in out}: out.append(e)
        # Two separate named entities need at least one legal minimum-duration cut each.
        need=float(self.cfg.min_visual_clip_seconds)*len(out)
        return out if len(out)>=2 and req.duration>=need-0.03 else []

    def _pick_collective_segment(self,req,seg_idx,clips,clip_embs,ledger,retrieval_log,exclude_ids=None):
        entities=self._collective_entities(req)
        if not entities:
            return self._pick_for_segment(req,seg_idx,clips,clip_embs,ledger,retrieval_log,None,exclude_ids,allow_collective=False)
        selected=[]; missing=[]; child_decisions=[]
        per=max(float(self.cfg.min_visual_clip_seconds),req.duration/max(1,len(entities)))
        for n,entity in enumerate(entities,1):
            # Keep semantic concepts, but narrow the hard requirement to one named
            # entity. Framing/camera preferences remain soft and cannot cause a reject.
            attrs=[a for a in req.required_attributes if entity.casefold() in a.casefold()]
            sub=replace(req,
                segment_id=f'{req.segment_id}__entity_{n:02d}',
                requested_visual=f'{entity} shown clearly in real source footage',
                primary_subject=entity,
                required_entities=[entity],
                required_attributes=attrs,
                timeline_start=0.0,timeline_end=per)
            chosen,dec=self._pick_for_segment(sub,seg_idx,clips,clip_embs,ledger,retrieval_log,None,exclude_ids,allow_collective=False,allow_gap_fill=False)
            child_decisions.append({'entity':entity,'decision':dec})
            if not chosen:
                missing.append(entity); continue
            # Usually this is one clip because per is only the entity's share of the
            # segment. If a short clip requires another, both remain valid/unique.
            for x in chosen:
                x['coverage_entity']=entity
                selected.append(x)
        remain=max(0.0,req.duration-sum(float(x.get('take',0.0)) for x in selected))
        # If a named sub-entity cannot be found but at least one legal cut remains,
        # fill the remainder with generally related unused B-roll before accepting a gap.
        if remain>=self.cfg.min_visual_clip_seconds-0.03:
            filler=self._relaxed_gap_fill(req,seg_idx,clips,clip_embs,ledger,retrieval_log,remain,{x['clip'].clip_id for x in selected},repeat=False)
            selected.extend(filler); remain=max(0.0,req.duration-sum(float(x.get('take',0.0)) for x in selected))
        unused_legal=self._unused_legal_count(clips,ledger)
        if remain>=self.cfg.min_visual_clip_seconds-0.03 and unused_legal < int(self.cfg.min_unused_clips_before_repeat):
            filler=self._relaxed_gap_fill(req,seg_idx,clips,clip_embs,ledger,retrieval_log,remain,{x['clip'].clip_id for x in selected},repeat=True)
            selected.extend(filler); remain=max(0.0,req.duration-sum(float(x.get('take',0.0)) for x in selected))
        if not selected:
            return [],{'segment_id':req.segment_id,'status':'no_match','reason':'No acceptable or generally related footage was found for this multi-entity segment.','requested_visual':req.requested_visual,'collective_entities':entities,'missing_entities':missing,'entity_decisions':child_decisions}
        dec={'segment_id':req.segment_id,'status':'matched','requested_visual':req.requested_visual,'coverage_shortfall':round(remain,3),'collective_entities':entities,'missing_entities':missing,'entity_decisions':child_decisions,'selected':[{'clip_id':x['clip'].clip_id,'coverage_entity':x.get('coverage_entity'),'decision':x['judgment'].decision,'reason':x['judgment'].reason,'confidence':x['confidence'],'use_number':x['use_number'],'gap_filler':bool(x.get('gap_filler'))} for x in selected]}
        if missing: dec['warning']='Missing footage for: '+', '.join(missing)
        return selected,dec

    def _pick_for_segment(self,req,seg_idx,clips,clip_embs,ledger,retrieval_log,req_emb=None,exclude_ids=None,allow_collective=True,allow_gap_fill=True):
        if bool(getattr(self.cfg,'editorial_freedom_enabled',True)) and getattr(req,'match_mode','thematic')=='brand_filler':
            return self._pick_brand_filler(req,seg_idx,clips,clip_embs,ledger,retrieval_log,req_emb,exclude_ids)
        if allow_collective and self._collective_entities(req):
            return self._pick_collective_segment(req,seg_idx,clips,clip_embs,ledger,retrieval_log,exclude_ids)
        if req_emb is None:
            req_emb=embed(self.cfg.ollama_url,self.cfg.embedding_model,[req_text(req)])[0]
        usage={c.clip_id:ledger.info(c.clip_id) for c in clips}
        ranked=rank(req,clips,req_emb,clip_embs,usage)
        ranked=self._apply_visual_diversity(ranked,clips,clip_embs,ledger,seg_idx,req)
        excluded=set(exclude_ids or [])
        if excluded: ranked=[c for c in ranked if c.clip.clip_id not in excluded]
        face_free,checked=self._face_free_shortlist(ranked)
        retrieval_log.append({'segment_id':req.segment_id,'requested_visual':req.requested_visual,'match_mode':getattr(req,'match_mode','thematic'),'context_anchor':getattr(req,'context_anchor',''),'candidates':[dict(c.compact_dict(),face_status=c.face_status,face_reason=c.face_reason) for c in checked]})
        if not face_free:
            return [],{'segment_id':req.segment_id,'status':'no_match','reason':'No allowed candidate survived Agent 1 face metadata filtering (visible faces are rejected; uncertain is allowed only when configured).','requested_visual':req.requested_visual}

        # Policy order is strict: all reasonable UNUSED footage is judged before any repeat.
        unused=[c for c in face_free if ledger.info(c.clip.clip_id)['count']==0 and c.retrieval_score>=self.cfg.min_unused_retrieval_score and c.clip.duration>=self.cfg.min_visual_clip_seconds-0.03]
        unused_pairs=self._judge_pool(req,unused,target_duration=req.duration,allow_weak=True) if unused else []
        selected=[]; remain=req.duration; used_here=set()

        def consume(pairs, allow_weak, repeat):
            nonlocal remain
            passes=(('strong_match',),('acceptable_match',),('weak_match',) if allow_weak else tuple())
            for allowed in passes:
                if not allowed: continue
                for cand,j in pairs:
                    if remain<=0.03: return
                    cid=cand.clip.clip_id
                    if cid in used_here or j.decision not in allowed or j.hard_requirement_failed: continue
                    threshold=self.cfg.min_repeat_retrieval_score if repeat else self.cfg.min_unused_retrieval_score
                    if cand.retrieval_score < threshold: continue
                    if j.decision=='weak_match' and req.specificity in {'strict','high'}: continue
                    if not ledger.can_use(cid,seg_idx,repeat=repeat): continue
                    use_no=ledger.info(cid)['count']+1
                    take=self._planned_take(remain,cand.clip.duration)
                    if take<=0.03: continue
                    ss,se=choose_window(cand.clip,take,use_no)
                    conf=max(0.0,min(1.0,.65*LEVEL_BASE[j.decision]+.35*cand.retrieval_score))
                    selected.append({'clip':cand.clip,'judgment':j,'source_start':ss,'source_end':se,'take':take,'confidence':round(conf,3),'use_number':use_no,'face_status':cand.face_status})
                    ledger.mark(cid,seg_idx); used_here.add(cid); remain-=take

        # Most relevant unused -> acceptable/close filler unused.
        consume(unused_pairs,allow_weak=True,repeat=False)

        # If a legal >=3s gap remains, broaden only the fine-detail requirements and
        # inspect at most two already-indexed unused candidates. This is intentionally
        # cheap and prevents black gaps without reopening an exhaustive search.
        if allow_gap_fill and remain>=self.cfg.min_visual_clip_seconds-0.03:
            filler=self._relaxed_gap_fill(req,seg_idx,clips,clip_embs,ledger,retrieval_log,remain,used_here|set(exclude_ids or []),repeat=False)
            selected.extend(filler); used_here.update(x['clip'].clip_id for x in filler); remain-=sum(float(x.get('take',0.0)) for x in filler)

        # Repeats are a true last resort. In a rich library, keep choosing different
        # topical shots even when they are less literal. This is the behavior a human
        # faceless-video editor would prefer over replaying the same B-roll.
        unused_legal=self._unused_legal_count(clips,ledger)
        repeat_allowed=unused_legal < int(self.cfg.min_unused_clips_before_repeat)
        if remain>0.03 and repeat_allowed:
            repeat_pool=[c for c in face_free if ledger.info(c.clip.clip_id)['count']==1 and c.retrieval_score>=self.cfg.min_repeat_retrieval_score and c.clip.duration>=self.cfg.min_visual_clip_seconds-0.03 and ledger.can_use(c.clip.clip_id,seg_idx,repeat=True)]
            repeat_pairs=self._judge_pool(req,repeat_pool,target_duration=remain,allow_weak=False) if repeat_pool else []
            consume(repeat_pairs,allow_weak=False,repeat=True)

        if allow_gap_fill and remain>=self.cfg.min_visual_clip_seconds-0.03 and repeat_allowed:
            filler=self._relaxed_gap_fill(req,seg_idx,clips,clip_embs,ledger,retrieval_log,remain,used_here|set(exclude_ids or []),repeat=True)
            selected.extend(filler); used_here.update(x['clip'].clip_id for x in filler); remain-=sum(float(x.get('take',0.0)) for x in filler)

        if not selected:
            best_decision=unused_pairs[0][1].decision if unused_pairs else 'none'
            return [],{'segment_id':req.segment_id,'status':'no_match','reason':f'No unused acceptable face-allowed footage remained; repeat policy also could not provide an allowed match. Best unused decision: {best_decision}.','requested_visual':req.requested_visual}
        decision={'segment_id':req.segment_id,'status':'matched','requested_visual':req.requested_visual,'coverage_shortfall':round(max(0,remain),3),'selected':[{'clip_id':x['clip'].clip_id,'decision':x['judgment'].decision,'reason':x['judgment'].reason,'confidence':x['confidence'],'use_number':x['use_number'],'gap_filler':bool(x.get('gap_filler'))} for x in selected]}
        if remain>0.03: decision['warning']=f'Coverage shortfall {remain:.2f}s after exhausting allowed relevant clips.'
        return selected,decision

    def run(self,script:Path,audio:Path,agent1_ws:Path,force=False,skip_global=False):
        self.ws.mkdir(parents=True,exist_ok=True)
        qc1p=agent1_ws/'qc_report.json'
        if not qc1p.exists(): raise FileNotFoundError(f'Agent 1 QC report not found: {qc1p}')
        qc1=read_json(qc1p)
        if not qc1.get('passed',False): raise RuntimeError('Agent 1 workspace QC is not passing; complete/fix indexing before Agent 2.')
        clips=load_agent1_clips(agent1_ws)
        if not clips: raise RuntimeError('Agent 1 contains no indexed clips')
        face_counts={'no_face':0,'face_visible':0,'uncertain':0}
        for c in clips: face_counts[c.face_status]=face_counts.get(c.face_status,0)+1
        eligible_clips=[c for c in clips if c.face_status=='no_face' or (c.face_status=='uncertain' and self.cfg.allow_uncertain_face_status)]
        if not any(c.usable for c in eligible_clips): raise RuntimeError('No usable clips remain after Agent 1 face metadata filtering.')
        missing_sources=sorted({c.source_path for c in eligible_clips if c.usable and not Path(c.source_path).exists()})
        if missing_sources: raise FileNotFoundError('Indexed eligible source footage is missing: '+', '.join(missing_sources[:5]))
        self.log.info(f"Loaded {len(clips)} indexed clips from Agent 1")
        self.log.info(f"Agent 1 face metadata: {face_counts}; {len(eligible_clips)} clip(s) eligible")
        editorial_profiled=sum(1 for c in eligible_clips if self._has_editorial_profile(c))
        if editorial_profiled:
            self.log.info(f"Agent 1 Editorial Vision metadata: {editorial_profiled}/{len(eligible_clips)} eligible clip(s) enhanced; legacy clips remain compatible")
        else:
            self.log.info(f"Agent 1 Editorial Vision metadata: 0/{len(eligible_clips)} eligible clip(s); running v1.0.19-compatible legacy selection")
        script_text=script.read_text(encoding='utf-8').strip()
        if not script_text: raise ValueError('Script is empty')
        sig={'script_sha256':sha256_file(script),'audio_sha256':sha256_file(audio),'agent1_db_sha256':sha256_file(agent1_ws/'footage_index.db'),'config_signature':self.cfg.signature()}
        statep=self.ws/'state.json'; old=read_json(statep) if statep.exists() else {}
        alignment_p=self.ws/'alignment.json'
        if not force and old.get('script_sha256')==sig['script_sha256'] and old.get('audio_sha256')==sig['audio_sha256'] and alignment_p.exists():
            alignment=read_json(alignment_p); self.log.info('Audio alignment unchanged -> reuse')
        else:
            self.log.info('Aligning script to voiceover with local whisper.cpp...')
            alignment=build_alignment(script,audio,self.ws,self.cfg,self.root); write_json(alignment_p,alignment)
        if alignment['alignment_ratio']<0.50: raise RuntimeError(f"Script/voiceover alignment too weak: {alignment['alignment_ratio']:.2f}")
        write_json(statep,sig|{'pipeline_version':'agent2-v1.0.20','planner_version':PLANNER_VERSION,'alignment_ready':True,'visual_segments_ready':bool(old.get('visual_segments_ready',False) and old.get('planner_version')==PLANNER_VERSION and old.get('script_sha256')==sig['script_sha256'] and old.get('audio_sha256')==sig['audio_sha256'])})
        words=alignment['words']
        segments_p=self.ws/'visual_segments.json'
        if not force and old.get('planner_version')==PLANNER_VERSION and old.get('script_sha256')==sig['script_sha256'] and old.get('audio_sha256')==sig['audio_sha256'] and segments_p.exists():
            seg_data=read_json(segments_p); planned_reqs=[VisualRequirement(**x) for x in seg_data['segments']]; self.log.info('Visual segments unchanged -> reuse')
        else:
            self.log.info('Planning meaningful visual segments + editorial match modes with Qwen3.5...')
            planned_reqs=plan_segments(self.cfg,script_text,words,alignment['audio_duration']); write_json(segments_p,{'segments':[r.to_dict() for r in planned_reqs]})
        # v1.0.15 separates what the narration means from how strictly footage must
        # illustrate it. The effective requirements below are what retrieval sees.
        reqs=[self._editorial_requirement(r) for r in planned_reqs]
        mode_counts={m:sum(1 for r in reqs if getattr(r,'match_mode','thematic')==m) for m in ('literal','thematic','brand_filler')}
        write_json(self.ws/'editorial_plan.json',{'planner_version':PLANNER_VERSION,'match_policy_version':MATCH_POLICY_VERSION,'mode_counts':mode_counts,'segments':[{'segment_id':r.segment_id,'match_mode':getattr(r,'match_mode','thematic'),'context_anchor':getattr(r,'context_anchor',''),'narration_text':r.narration_text,'planned_requested_visual':p.requested_visual,'effective_requested_visual':r.requested_visual,'planned_required_entities':p.required_entities,'effective_required_entities':r.required_entities} for p,r in zip(planned_reqs,reqs)]})
        write_json(statep,sig|{'pipeline_version':'agent2-v1.0.20','planner_version':PLANNER_VERSION,'alignment_ready':True,'visual_segments_ready':True})
        self.log.info(f"Visual segments: {len(reqs)} | editorial modes: literal={mode_counts['literal']}, thematic={mode_counts['thematic']}, brand_filler={mode_counts['brand_filler']}")
        # Face-visible clips were discarded immediately after loading Agent 1 metadata.
        # Agent 2 performs no frame extraction and no vision-model face scan.
        clip_embs=self._ensure_embeddings(eligible_clips)
        req_embs={}
        for req in reqs:
            req_embs[req.segment_id]=embed(self.cfg.ollama_url,self.cfg.embedding_model,[req_text(req)])[0]
        ledger=UsageLedger(self.cfg.max_clip_uses,self.cfg.min_repeat_segment_gap)
        retrieval_log=[]; decisions=[]; timeline_segments=[]; clip_map={c.clip_id:c for c in clips}
        progress_p=self.ws/'matching_progress.json'
        match_key=sha256_text(json.dumps(sig,sort_keys=True)+sha256_text(json.dumps([r.to_dict() for r in reqs],sort_keys=True))+MATCH_POLICY_VERSION+RERANK_VERSION)
        start_idx=0
        if not force and progress_p.exists():
            prog=read_json(progress_p)
            if prog.get('match_key')==match_key:
                timeline_segments=prog.get('timeline_segments',[]); decisions=prog.get('decisions',[]); retrieval_log=prog.get('retrieval_log',[])
                start_idx=len(timeline_segments)
                if start_idx>len(reqs): start_idx=0; timeline_segments=[]; decisions=[]; retrieval_log=[]
                for si,seg in enumerate(timeline_segments):
                    for tc in seg.get('clips',[]): ledger.mark(tc['clip_id'],si)
                if start_idx: self.log.info(f'Resuming matching at segment {start_idx+1}; {start_idx} completed segment(s) preserved')
        planned_by={r.segment_id:r for r in planned_reqs}
        for i,req in enumerate(reqs):
            if i<start_idx: continue
            planned=planned_by.get(req.segment_id,req)
            mode=getattr(req,'match_mode','thematic'); anchor=getattr(req,'context_anchor','')
            self.log.info(f"Matching {req.segment_id} [{mode}{' | '+anchor if anchor else ''}]: {req.requested_visual}")
            chosen,dec=self._pick_for_segment(req,i,eligible_clips,clip_embs,ledger,retrieval_log,req_embs.get(req.segment_id)); decisions.append(dec)
            common={'segment_id':req.segment_id,'timeline_start':round(req.timeline_start,3),'timeline_end':round(req.timeline_end,3),
                    'narration_text':req.narration_text,'match_mode':mode,'context_anchor':anchor,
                    'planned_requested_visual':planned.requested_visual,'requested_visual':req.requested_visual,
                    'planned_required_entities':planned.required_entities,'required_entities':req.required_entities}
            if not chosen:
                timeline_segments.append(common|{'status':'no_match','reason':dec['reason'],'clips':[]})
            else:
                cur=req.timeline_start; tclips=[]
                for x in chosen:
                    dur=x['source_end']-x['source_start']; end=min(req.timeline_end,cur+dur)
                    tclips.append({'clip_id':x['clip'].clip_id,'source':x['clip'].source_path,'source_video':x['clip'].source_video,'source_start':round(x['source_start'],3),'source_end':round(x['source_start']+(end-cur),3),'timeline_start':round(cur,3),'timeline_end':round(end,3),'reason':x['judgment'].reason,'match_level':x['judgment'].decision,'confidence':x['confidence'],'use_number':x['use_number'],'face_status':x['face_status'],'coverage_entity':x.get('coverage_entity'),'gap_filler':bool(x.get('gap_filler')),'editorial_filler':bool(x.get('editorial_filler')),'agent1_editorial_profile_version':getattr(x['clip'],'editorial_profile_version',0),'brand':getattr(x['clip'],'brand','unknown'),'model_name':getattr(x['clip'],'model_name','unknown'),'subject_focus':getattr(x['clip'],'subject_focus','unknown'),'editorial_role':getattr(x['clip'],'editorial_role','other'),'visual_family':getattr(x['clip'],'visual_family','unknown'),'visual_quality_score':round(float(getattr(x['clip'],'visual_quality_score',0.5)),3),'editorial_usefulness_score':round(float(getattr(x['clip'],'editorial_usefulness_score',0.5)),3)})
                    cur=end
                    if cur>=req.timeline_end-0.03: break
                timeline_segments.append(common|{'status':'matched','coverage_shortfall':round(max(0,req.timeline_end-cur),3),'clips':tclips})
            write_json(progress_p,{'match_key':match_key,'timeline_segments':timeline_segments,'decisions':decisions,'retrieval_log':retrieval_log})
        (self.ws/'retrieval_candidates.jsonl').write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in retrieval_log)+'\n',encoding='utf-8')
        write_json(self.ws/'match_decisions.json',{'decisions':decisions}); write_json(self.ws/'no_matches.json',{'no_matches':[d for d in decisions if d['status']=='no_match']})
        timeline={'schema_version':1,'project':self.ws.name,'audio':{'source':str(audio.resolve()),'duration':alignment['audio_duration']},'policy':{'faces':'agent1_metadata_reject_face_visible_allow_uncertain','face_source':'agent1_persistent_metadata','usage':'unused relevant -> unused close/filler -> repeat relevant','max_uses_per_clip':self.cfg.max_clip_uses,'min_repeat_segment_gap':self.cfg.min_repeat_segment_gap,'clip_duration_seconds':{'min':self.cfg.min_visual_clip_seconds,'max':self.cfg.max_visual_clip_seconds,'prefer_longer':self.cfg.prefer_longer_clips},'editorial_freedom':'literal beats seek exact footage; thematic beats use broad topic relevance; brand_filler beats intentionally use varied same-brand/same-topic footage without sentence-level literal matching','editorial_modes':mode_counts,'gap_fill':'if literal/thematic coverage fails, relax to context-aware brand/topic filler before any repeat','diversity':'unused-first plus source rotation, source-time near-duplicate avoidance, embedding similarity penalties and source balancing','editorial_vision_adapter':'Agent 1 v1.3 visual_family/role/focus/brand/model/quality metadata is used when present; legacy clips remain fully compatible'},'segments':timeline_segments}
        review_issues=[]
        risks=review_risks(timeline_segments)
        if self.cfg.global_review_enabled and not skip_global and risks:
            self.log.info('Running global editorial review for risk(s): '+', '.join(risks))
            risk_segment_ids={x.split(':',1)[0] for x in risks}
            compact=[{'segment_id':s['segment_id'],'narration_text':s['narration_text'],'requested_visual':s['requested_visual'],'required_entities':s.get('required_entities',[]),'status':s['status'],'clips':[{'clip_id':c['clip_id'],'source_video':c['source_video'],'reason':c['reason'],'match_level':c.get('match_level'),'use_number':c['use_number'],'coverage_entity':c.get('coverage_entity')} for c in s['clips']]} for s in timeline_segments if s['segment_id'] in risk_segment_ids]
            review_issues=global_review(self.cfg,compact)
        elif self.cfg.global_review_enabled and not skip_global:
            self.log.info('Global editorial review skipped: no weak or repeated selections detected')

        if review_issues:
            req_by={r.segment_id:(i,r) for i,r in enumerate(reqs)}
            rematched=set()
            for issue in review_issues:
                sid=issue.get('segment_id'); action=issue.get('action'); severity=issue.get('severity')
                if severity!='major' or action=='keep' or sid not in req_by or sid in rematched: continue
                idx,req=req_by[sid]; seg=timeline_segments[idx]; old_ids=[c['clip_id'] for c in seg.get('clips',[])]
                for cid in old_ids: ledger.release(cid,idx)
                if action=='no_match':
                    planned=planned_by.get(req.segment_id,req)
                    timeline_segments[idx]={'segment_id':req.segment_id,'timeline_start':round(req.timeline_start,3),'timeline_end':round(req.timeline_end,3),'narration_text':req.narration_text,'match_mode':getattr(req,'match_mode','thematic'),'context_anchor':getattr(req,'context_anchor',''),'planned_requested_visual':planned.requested_visual,'requested_visual':req.requested_visual,'planned_required_entities':planned.required_entities,'required_entities':req.required_entities,'status':'no_match','reason':'Global editorial review rejected the prior selection: '+issue.get('issue',''),'clips':[]}
                    decisions[idx]={'segment_id':req.segment_id,'status':'no_match','requested_visual':req.requested_visual,'reason':timeline_segments[idx]['reason']}; issue['resolved']=True; issue['resolution']='converted_to_no_match'; rematched.add(sid); continue
                if action=='rematch' and self.cfg.max_global_rematches>0:
                    self.log.warning(f'Global review rematching {sid}: {issue.get("issue","")}')
                    chosen,dec=self._pick_for_segment(req,idx,eligible_clips,clip_embs,ledger,retrieval_log,req_embs.get(req.segment_id),exclude_ids=old_ids)
                    decisions[idx]=dec
                    if not chosen:
                        planned=planned_by.get(req.segment_id,req)
                        timeline_segments[idx]={'segment_id':req.segment_id,'timeline_start':round(req.timeline_start,3),'timeline_end':round(req.timeline_end,3),'narration_text':req.narration_text,'match_mode':getattr(req,'match_mode','thematic'),'context_anchor':getattr(req,'context_anchor',''),'planned_requested_visual':planned.requested_visual,'requested_visual':req.requested_visual,'planned_required_entities':planned.required_entities,'required_entities':req.required_entities,'status':'no_match','reason':'Global review requested rematch, but no alternate acceptable face-allowed footage was available.','clips':[]}
                        issue['resolved']=True; issue['resolution']='rematch_fell_back_to_no_match'
                    else:
                        cur=req.timeline_start; tclips=[]
                        for x in chosen:
                            dur=x['source_end']-x['source_start']; end=min(req.timeline_end,cur+dur)
                            tclips.append({'clip_id':x['clip'].clip_id,'source':x['clip'].source_path,'source_video':x['clip'].source_video,'source_start':round(x['source_start'],3),'source_end':round(x['source_start']+(end-cur),3),'timeline_start':round(cur,3),'timeline_end':round(end,3),'reason':x['judgment'].reason,'match_level':x['judgment'].decision,'confidence':x['confidence'],'use_number':x['use_number'],'face_status':x['face_status'],'coverage_entity':x.get('coverage_entity'),'gap_filler':bool(x.get('gap_filler')),'editorial_filler':bool(x.get('editorial_filler')),'agent1_editorial_profile_version':getattr(x['clip'],'editorial_profile_version',0),'brand':getattr(x['clip'],'brand','unknown'),'model_name':getattr(x['clip'],'model_name','unknown'),'subject_focus':getattr(x['clip'],'subject_focus','unknown'),'editorial_role':getattr(x['clip'],'editorial_role','other'),'visual_family':getattr(x['clip'],'visual_family','unknown'),'visual_quality_score':round(float(getattr(x['clip'],'visual_quality_score',0.5)),3),'editorial_usefulness_score':round(float(getattr(x['clip'],'editorial_usefulness_score',0.5)),3)})
                            cur=end
                            if cur>=req.timeline_end-0.03: break
                        planned=planned_by.get(req.segment_id,req)
                        timeline_segments[idx]={'segment_id':req.segment_id,'timeline_start':round(req.timeline_start,3),'timeline_end':round(req.timeline_end,3),'narration_text':req.narration_text,'match_mode':getattr(req,'match_mode','thematic'),'context_anchor':getattr(req,'context_anchor',''),'planned_requested_visual':planned.requested_visual,'requested_visual':req.requested_visual,'planned_required_entities':planned.required_entities,'required_entities':req.required_entities,'status':'matched','coverage_shortfall':round(max(0,req.timeline_end-cur),3),'clips':tclips,'rematched_by_global_review':True}
                        issue['resolved']=True; issue['resolution']='rematched'
                    rematched.add(sid)
            # Recompute displayed use numbers after any global-review mutation.
            seen={}
            for seg in timeline_segments:
                for tc in seg.get('clips',[]):
                    seen[tc['clip_id']]=seen.get(tc['clip_id'],0)+1; tc['use_number']=seen[tc['clip_id']]
            write_json(progress_p,{'match_key':match_key,'timeline_segments':timeline_segments,'decisions':decisions,'retrieval_log':retrieval_log})
        timeline['segments']=timeline_segments
        timeline['editorial_review']=review_issues
        diversity=self._diversity_metrics(timeline_segments,clip_map,clip_embs)
        timeline['diversity']=diversity
        write_json(self.ws/'timeline.json',timeline)
        qc=validate_timeline(timeline,clip_map,self.cfg.max_clip_uses,self.cfg.min_visual_clip_seconds,self.cfg.max_visual_clip_seconds)
        qc['diversity']=diversity
        majors=unresolved_major_issues(review_issues)
        if majors: qc['passed']=False; qc['issues'].extend([f"Editorial review {x['segment_id']}: {x['issue']}" for x in majors])
        qc['alignment_ratio']=alignment['alignment_ratio']; qc['face_policy']='agent1_metadata_reject_visible_allow_uncertain' if self.cfg.allow_uncertain_face_status else 'agent1_metadata_strict_no_face_only'; write_json(self.ws/'qc_report.json',qc)
        filler_segments=sum(1 for s in timeline_segments if s.get('match_mode')=='brand_filler')
        literal_fallback_segments=sum(1 for s in timeline_segments if s.get('match_mode')=='literal' and any(c.get('gap_filler') for c in s.get('clips',[])))
        if not qc['passed']:
            for issue in qc.get('issues',[]):
                self.log.warning('QC | %s',issue)
        summary={'segments_total':len(reqs),'segments_matched':qc['matched_segments'],'segments_no_match':qc['no_match_segments'],'editorial_modes':mode_counts,'brand_filler_segments':filler_segments,'literal_segments_using_context_fallback':literal_fallback_segments,'unique_clips_used':len(qc['clip_use_counts']),'repeated_clips':sum(1 for n in qc['clip_use_counts'].values() if n>1),'nearby_source_region_reuses':diversity['nearby_source_region_reuses'],'semantic_near_duplicate_reuses':diversity['semantic_near_duplicate_reuses'],'same_source_consecutive_cuts':diversity['same_source_consecutive_cuts'],'visual_family_reuses':diversity['visual_family_reuses'],'same_editorial_role_focus_consecutive':diversity['same_editorial_role_focus_consecutive'],'agent1_editorial_profile_eligible':editorial_profiled,'agent1_editorial_profile_clips_used':diversity['agent1_editorial_profile_clips_used'],'unique_visual_families_used':diversity['unique_visual_families_used'],'visual_diversity_score':diversity['visual_diversity_score'],'source_distribution':diversity['source_distribution'],'uncertain_face_clips_used':qc.get('uncertain_face_clips_used',0),'qc_issues':qc.get('issues',[]),'qc_passed':qc['passed']}
        write_json(self.ws/'run_summary.json',summary); write_json(statep,sig|{'pipeline_version':'agent2-v1.0.20','planner_version':PLANNER_VERSION,'alignment_ready':True,'visual_segments_ready':True,'timeline_ready':True,'qc_passed':qc['passed']})
        return summary
