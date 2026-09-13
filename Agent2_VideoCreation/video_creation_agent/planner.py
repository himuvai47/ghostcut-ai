from __future__ import annotations
import re
import logging
from dataclasses import replace
from .models import VisualRequirement,VISUAL_INTENTS,SPECIFICITY,MATCH_MODES
from .ollama import chat

PLANNER_VERSION='planner-v13-compact-editorial-planner'
SHOT_TYPES=["extreme_close_up","macro","close_up","medium_close_up","medium","medium_wide","wide","overhead","pov","insert","text_screen","unknown"]
CONTENT_TYPES=["product_closeup","product_in_box","product_handling","presenter_demo","product_detail","talking_head","environment","action","text_screen","intro_card","end_screen","credits","logo_screen","other"]

log=logging.getLogger(__name__)


_CONTENT_TYPE_ALIASES={
    'comparison':['product_closeup','product_detail'],
    'product_comparison':['product_closeup','product_detail'],
    'beauty_shot':['product_closeup'],
    'product_beauty':['product_closeup'],
    'wrist_shot':['product_handling'],
    'wrist':['product_handling'],
    'movement':['product_detail'],
    'movement_detail':['product_detail'],
    'macro_detail':['product_detail'],
    'detail':['product_detail'],
    'retail_display':['environment','product_closeup'],
    'boutique':['environment'],
    'store':['environment'],
    'forum':['text_screen'],
    'screenshot':['text_screen'],
    'brand_logo':['logo_screen'],
}
_SHOT_TYPE_ALIASES={
    'closeup':'close_up','close-up':'close_up','close up':'close_up',
    'extreme close-up':'extreme_close_up','extreme close up':'extreme_close_up',
    'medium close-up':'medium_close_up','medium close up':'medium_close_up',
    'medium wide':'medium_wide','medium-wide':'medium_wide',
    'product shot':'close_up','detail':'close_up',
}

def _norm_label(value):
    return re.sub(r'[^a-z0-9]+','_',str(value or '').strip().casefold()).strip('_')

def _normalize_content_types(values):
    out=[]
    supported=set(CONTENT_TYPES)
    for raw in values or []:
        key=_norm_label(raw)
        mapped=_CONTENT_TYPE_ALIASES.get(key)
        if mapped is None and key in supported:
            mapped=[key]
        for item in mapped or []:
            if item in supported and item not in out:
                out.append(item)
    return out

def _normalize_shot_types(values):
    out=[]
    supported=set(SHOT_TYPES)
    for raw in values or []:
        key=_norm_label(raw)
        mapped=_SHOT_TYPE_ALIASES.get(key,key)
        if mapped in supported and mapped not in out:
            out.append(mapped)
    return out

def _numeric_word_index(raw):
    """Return the trailing integer in a model word-id variant, or None."""
    if raw is None:
        return None
    m=re.search(r'(\d+)\s*$',str(raw).strip())
    return int(m.group(1)) if m else None

def _resolve_model_word_id(raw,ids,pos):
    """Resolve harmless formatting variants without changing semantic boundaries.

    Whisper/alignment ids are canonical (for example w00042), while a long-form
    model occasionally returns w42, word_42, or drops zero padding.  Those all
    identify the same boundary and can be repaired deterministically.
    """
    if raw in pos:
        return raw
    text=str(raw or '').strip()
    # Case-only differences are harmless too.
    folded={x.casefold():x for x in ids}
    if text.casefold() in folded:
        return folded[text.casefold()]
    n=_numeric_word_index(text)
    if n is None:
        return None
    matches=[x for x in ids if _numeric_word_index(x)==n]
    return matches[0] if len(matches)==1 else None

def _repair_invalid_word_ids(segs,ids,pos):
    """Repair only nonexistent planner ids; keep valid range mistakes strict.

    Start boundaries are forced only when the model returned an unresolvable id,
    because contiguous coverage makes the expected next word deterministic.  An
    invalid end boundary is inferred from the next resolvable segment start, or
    the final script word for the last segment.  Valid ids are never moved here,
    so genuine gaps/overlaps still fail the normal validation below.
    """
    repaired=[]
    expected=0
    for i,orig in enumerate(segs):
        s=dict(orig)
        raw_start=s.get('start_word_id'); raw_end=s.get('end_word_id')
        start=_resolve_model_word_id(raw_start,ids,pos)
        if start is None:
            if expected>=len(ids):
                raise ValueError('Planner returned invalid start word id beyond script end')
            start=ids[expected]
            log.warning('Planner returned invalid start_word_id %r; repaired to %s',raw_start,start)
        s['start_word_id']=start

        end=_resolve_model_word_id(raw_end,ids,pos)
        if end is None:
            inferred=None
            # The next valid start uniquely determines this end boundary.
            for j in range(i+1,len(segs)):
                nxt=_resolve_model_word_id(segs[j].get('start_word_id'),ids,pos)
                if nxt is not None:
                    ni=pos[nxt]
                    if ni>pos[start]:
                        inferred=ids[ni-1]
                    break
            if inferred is None and i==len(segs)-1:
                inferred=ids[-1]
            if inferred is None:
                raise ValueError(f'Planner returned invalid end_word_id {raw_end!r} and no safe boundary could be inferred')
            end=inferred
            log.warning('Planner returned invalid end_word_id %r; repaired to %s',raw_end,end)
        s['end_word_id']=end
        repaired.append(s)

        # Advance only from the resolved boundary. Any semantic gap/overlap is still
        # caught later; this value is used solely to repair a following invalid start.
        if end in pos:
            expected=pos[end]+1
    return repaired


def _repair_planner_ranges(segs, ids, pos):
    """Stitch planner word ranges into exact contiguous script coverage.

    Qwen can return semantically useful ordered segments whose numeric boundaries
    contain a one-word gap, overlap, or small ordering mistake.  Those boundary
    defects are mechanical, not editorial, so rejecting the whole long-form plan
    wastes an otherwise good model response.

    This pass preserves segment count/order/content/match_mode and adjusts only
    word boundaries.  Each interior cut is placed halfway between the left
    segment's proposed end+1 and the right segment's proposed start, then clamped
    so every segment owns at least one real aligned word.  The first segment is
    anchored to the first script word and the final segment to the last word.
    """
    if not segs:
        return segs
    n_words=len(ids)
    n_segs=len(segs)
    if n_segs>n_words:
        raise ValueError(f'Planner returned {n_segs} segments for only {n_words} aligned words')

    starts=[pos[s['start_word_id']] for s in segs]
    ends=[pos[s['end_word_id']] for s in segs]
    cuts=[]
    prev=0
    for i in range(n_segs-1):
        # cut is the first word owned by the next segment.  Averaging the two
        # proposed sides treats gaps/overlaps symmetrically instead of blindly
        # trusting one malformed boundary.
        left_cut=ends[i]+1
        right_cut=starts[i+1]
        target=(left_cut+right_cut)//2
        lower=prev+1
        remaining=n_segs-(i+1)
        upper=n_words-remaining
        cut=max(lower,min(target,upper))
        cuts.append(cut)
        prev=cut

    bounds=[]
    start=0
    for i in range(n_segs):
        end=(cuts[i]-1) if i<len(cuts) else n_words-1
        bounds.append((start,end))
        start=end+1

    out=[]
    changed=False
    for i,(orig,(a,b)) in enumerate(zip(segs,bounds)):
        s=dict(orig)
        old_a=pos[s['start_word_id']]; old_b=pos[s['end_word_id']]
        if old_a!=a or old_b!=b:
            changed=True
            log.warning(
                'Planner range repair segment %d: %s..%s -> %s..%s',
                i+1,s['start_word_id'],s['end_word_id'],ids[a],ids[b])
        s['start_word_id']=ids[a]
        s['end_word_id']=ids[b]
        out.append(s)

    if changed:
        log.warning('Planner word ranges were repaired deterministically to contiguous full-script coverage')
    return out
SCHEMA={"type":"object","properties":{"segments":{"type":"array","items":{"type":"object","properties":{
"start_word_id":{"type":"string"},"end_word_id":{"type":"string"},"visual_intent":{"type":"string","enum":list(VISUAL_INTENTS)},"requested_visual":{"type":"string"},"primary_subject":{"type":"string"},"required_entities":{"type":"array","items":{"type":"string"}},"required_attributes":{"type":"array","items":{"type":"string"}},"preferred_shot_types":{"type":"array","items":{"type":"string"}},"preferred_content_types":{"type":"array","items":{"type":"string"}},"visual_concepts":{"type":"array","items":{"type":"string"}},"avoid":{"type":"array","items":{"type":"string"}},"specificity":{"type":"string","enum":list(SPECIFICITY)},"match_mode":{"type":"string","enum":list(MATCH_MODES)}},"required":["start_word_id","end_word_id","visual_intent","requested_visual","primary_subject","required_entities","required_attributes","preferred_shot_types","preferred_content_types","visual_concepts","avoid","specificity"],"additionalProperties":False}}},"required":["segments"],"additionalProperties":False}

_PRODUCTION_TERMS=(
    'panning','pan shot','tracking shot','dolly','zoom','x-ray','xray','animation','animated',
    'split screen','split-screen','montage','cutaway','transition','speed ramp','motion graphic',
    'extreme close-up','extreme close up','close-up','close up','macro shot','medium close-up',
    'medium close up','wide shot','overhead shot','static shot','static close-up','static close up'
)

# requested_visual is the semantic search target. Exact framing belongs in the
# preferred_* fields so it cannot accidentally become a hard rejection reason.
_FRAMING_PREFIXES=(
    r'extreme\s+close[- ]up(?:\s+macro)?\s+(?:shot|footage|view)?\s*(?:of)?\s*',
    r'(?:static\s+)?close[- ]up\s+(?:shot|footage|view)?\s*(?:of)?\s*',
    r'macro\s+(?:shot|footage|view)\s+(?:of)?\s*',
    r'medium\s+close[- ]up\s+(?:shot|footage|view)?\s*(?:of)?\s*',
    r'wide\s+(?:shot|footage|view)\s+(?:of)?\s*',
    r'overhead\s+(?:shot|footage|view)\s+(?:of)?\s*',
    r'static\s+(?:shot|footage|view)\s+(?:of)?\s*',
)

def _word_text(words): return ' '.join(f"{w['id']}={w['text']}" for w in words)

def _strip_framing_prefix(text:str)->str:
    s=text
    changed=True
    while changed:
        changed=False
        for pat in _FRAMING_PREFIXES:
            ns=re.sub(r'(?i)^\s*'+pat,'',s,count=1)
            if ns!=s:
                s=ns; changed=True
    return s

def _semanticize_requested_visual(text:str,narration:str)->str:
    """Turn a model-written shot direction into a semantic footage requirement."""
    s=' '.join((text or '').split())
    nl=narration.casefold()
    # Exact camera/framing words are never hard requirements here. The planner may
    # still return preferred_shot_types, which the matcher treats as soft signals.
    s=_strip_framing_prefix(s)
    if 'x-ray' not in nl and 'xray' not in nl and 'animation' not in nl and 'cutaway' not in nl:
        s=re.sub(r'(?i)\btransparent\s+x-?ray(?:\s+style)?\s+animation\s+or\s+cutaway\s+view\s+of\s+', '', s)
        s=re.sub(r'(?i)\bx-?ray(?:\s+style)?\s+animation\s+(?:of|showing)\s+', '', s)
        s=re.sub(r'(?i)\bcutaway\s+view\s+of\s+', '', s)
        s=re.sub(r'(?i)\banimat(?:ed|ion)\s+(?:view\s+)?(?:of|showing)\s+', '', s)
    if 'pan' not in nl:
        s=re.sub(r'(?i)\bpanning\s+shot\s+(?:across|of)\s+', 'view of ', s)
        s=re.sub(r'(?i)\bpan(?:ning)?\s+(?:across|over)\s+', 'view of ', s)
    if 'zoom' not in nl:
        s=re.sub(r'(?i)\bzoom(?:ing)?\s+(?:in|out)?\s*(?:on|into|toward|across)?\s*', '', s)
    # A comparison can be fulfilled by multiple ordinary source clips. Never make
    # split-screen/montage editing style part of the footage requirement.
    s=re.sub(r'(?i)^\s*split[- ](?:screen|view)\s+or\s+(?:rapid\s+)?(?:montage|sequential\s+cuts?)\s+(?:contrasting|showing)\s+', '', s)
    s=re.sub(r'(?i)^\s*split[- ](?:screen|view)\s+or\s+side[- ]by[- ]side\s+(?:comparison\s+)?(?:of\s+)?', '', s)
    s=re.sub(r'(?i)^\s*split[- ](?:screen|view)\s+(?:contrasting|showing)\s+', '', s)
    s=re.sub(r'(?i)^\s*(?:rapid\s+)?montage\s+(?:of|showing)\s+', '', s)
    s=re.sub(r'(?i)^\s*sequential\s+cuts?\s+(?:of|showing)\s+', '', s)
    s=_strip_framing_prefix(s)
    s=re.sub(r'\s+', ' ', s).strip(' ,.-')
    return s or text

def _filter_required_attributes(attrs,narration):
    nl=narration.casefold(); out=[]
    for a in attrs or []:
        low=a.casefold()
        # Production/framing language should never become a hard attribute. Real
        # visible facts such as "blue dial" or "gears visible" remain untouched.
        if any(t in low for t in _PRODUCTION_TERMS) or re.search(r'\b(?:macro|framing|close[- ]up|wide shot|overhead|static|panning|pan|zoom|tracking)\b',low):
            continue
        out.append(a)
    return out

def _uniq(items):
    out=[]
    for x in items or []:
        if x not in out: out.append(x)
    return out


_GENERIC_SOFT_ENTITIES={
    'buyer','buyers','customer','customers','collector','collectors','community','presenter','person','people',
    'watch','watches','movement','movements','graphic','graphics','chart','charts','timeline','data',
    'certification','certification text','text','overlay','airport','store','department store','boutique'
}

# v1.0.15: editorial freedom classifies what kind of visual accuracy a beat really
# needs. Most commentary does not need literal illustration; the goal is to keep
# the topic/brand coherent while allowing strong, varied filler.
_GENERIC_CONTEXT_ENTITIES=_GENERIC_SOFT_ENTITIES | {
    'luxury watch','luxury watches','swiss watch','swiss watches','mechanical watch','mechanical watches',
    'watch buyer','watch buyers','serious buyer','serious buyers','brand','brands','product','products',
    'tell me','luxury counter','swiss watch brands','swiss brand','secondary market','market data',
    'comments','comment section','comment sections','like','channel','next brand'
}
_GENERIC_CONTEXT_PREFIXES=(
    'tell me','tell us','i read','read every','a like','like genuinely','the next brand',
    'luxury counter','at a luxury counter','secondary market data'
)
_DETAIL_CUES=(
    'caliber','calibre','movement','caseback','dial','bracelet','clasp','bezel','crown','screw','screws',
    'plate','gear','gears','escapement','rotor','chronograph','mechanism','component','components','finishing'
)

def _clean_entity(value):
    return ' '.join(str(value or '').split())

def _is_generic_context(value):
    low=_clean_entity(value).casefold()
    if (not low) or low in _GENERIC_CONTEXT_ENTITIES:
        return True
    if any(low.startswith(x) for x in _GENERIC_CONTEXT_PREFIXES):
        return True
    # These are editorial/narration concepts, not stable product/topic anchors.
    generic_tokens={'counter','comments','comment','channel','market data','price point'}
    return any(x in low for x in generic_tokens)

def _looks_specific_entity(value):
    text=_clean_entity(value)
    if _is_generic_context(text):
        return False
    low=text.casefold()
    toks=re.findall(r'[a-z0-9]+',low)
    # Model/reference names tend to carry digits, punctuation, or at least three
    # lexical tokens. Two-token company/brand names remain useful context but do
    # not automatically force a literal visual.
    return bool(re.search(r'\d',text) or '-' in text or len(toks)>=3)

def _infer_match_mode(raw, visual_intent, narration, entities, specificity):
    """Normalize planner mode with conservative editorial guardrails.

    A brand mention by itself is not literal. Literal mode is reserved for beats
    where an exact visible product/detail/comparison materially helps the edit.
    """
    raw=str(raw or '').strip().casefold()
    mode=raw if raw in MATCH_MODES else ''
    named=[_clean_entity(e) for e in (entities or []) if not _is_generic_context(e)]
    specific=[e for e in named if _looks_specific_entity(e)]
    nl=str(narration or '').casefold()
    detail_cue=any(c in nl for c in _DETAIL_CUES)
    intent=str(visual_intent or '')

    if not mode:
        if intent=='literal_product_detail' and (named or detail_cue):
            mode='literal'
        elif intent=='comparison' and len(named)>=2:
            mode='literal'
        elif intent=='process' and (named or detail_cue):
            mode='literal'
        elif intent=='literal_product' and specific:
            mode='literal'
        elif intent in {'conceptual','presenter_context','historical_context'}:
            mode='brand_filler'
        else:
            mode='thematic'

    # Guard against over-literal planning on commentary. A generic brand mention
    # or abstract claim should not force exact footage.
    if mode=='literal':
        literal_case=(
            (intent=='literal_product_detail' and (named or detail_cue)) or
            (intent=='comparison' and len(named)>=2) or
            (intent=='process' and (specific or detail_cue)) or
            (intent=='literal_product' and bool(specific))
        )
        if not literal_case:
            mode='brand_filler' if intent in {'conceptual','presenter_context','historical_context'} else ('thematic' if named else 'brand_filler')

    if mode=='brand_filler' and intent in {'literal_product_detail','comparison'} and specific:
        # If the planner explicitly described a concrete model/detail, keep at
        # least thematic pressure rather than discarding the subject entirely.
        mode='thematic'
    return mode

def _explicit_context_entities(req):
    return [_clean_entity(e) for e in (req.required_entities or []) if not _is_generic_context(e)]

def _assign_context_anchors(reqs):
    """Carry the dominant brand/topic through abstract filler beats.

    One-off model names do not hijack the context; entities repeated across the
    script can become a new local anchor (for example a comparison brand section).
    """
    counts={}; display={}; ordered=[]
    for r in reqs:
        for e in _explicit_context_entities(r):
            k=e.casefold(); counts[k]=counts.get(k,0)+1; display.setdefault(k,e)
            if k not in ordered: ordered.append(k)
    # Product essays usually establish the main brand early. Prefer the earliest
    # brand-like entity over a later comparison model that merely appears often.
    main_key=next((k for k in ordered if not _looks_specific_entity(display[k])), None)
    if main_key is None and counts:
        main_key=max(counts,key=counts.get)
    main=display.get(main_key,'') if main_key else ''
    stable={k for k,v in counts.items() if v>=2}
    current=main
    out=[]
    for r in reqs:
        explicit=_explicit_context_entities(r)
        explicit_key=explicit[0].casefold() if explicit else ''
        # The current beat should honor its explicit named subject even when that
        # subject is a one-off comparison. Only repeated entities persist into the
        # following abstract/filler beats.
        if r.match_mode=='literal' and explicit:
            anchor=' vs '.join(explicit[:2]) if len(explicit)>1 else explicit[0]
        elif explicit:
            anchor=explicit[0]
        else:
            anchor=current or main
            if not anchor and not _is_generic_context(r.primary_subject):
                anchor=_clean_entity(r.primary_subject)
        if explicit and (explicit_key in stable or not current):
            current=explicit[0]
        out.append(replace(r,context_anchor=anchor))
    return out

def _explicit_named_entities(parent, narration):
    """Keep hard entities only when the split beat explicitly names them.

    Long parent segments often mention several brands/objects. Copying all of those
    entities into every 8-second child makes matching far too strict. Generic roles
    such as "buyer" or "presenter" are always soft B-roll concepts, never hard
    entities for a faceless rough cut.
    """
    nl=' '.join(str(narration or '').casefold().split())
    out=[]
    for e in parent.required_entities or []:
        clean=' '.join(str(e or '').split())
        low=clean.casefold()
        if not clean or low in _GENERIC_SOFT_ENTITIES:
            continue
        if low in nl:
            out.append(clean)
    return _uniq(out)

def _derive_split_child(parent, chunk_words):
    """Create a narration-specific semantic target for one paced child beat.

    This is deterministic and costs zero extra model calls. The narration itself is
    the best retrieval signal after an oversized semantic unit has been split.
    """
    narration=' '.join(w['text'] for w in chunk_words).strip()
    entities=_explicit_named_entities(parent,narration)
    preview=' '.join(narration.split()[:34]).strip()
    if entities:
        requested=f"Real footage of {', '.join(entities)} relevant to: {preview}"
        primary=entities[0]
    else:
        subject=' '.join(str(parent.primary_subject or '').split())
        prefix=f"Relevant real B-roll of {subject}" if subject else 'Relevant real topical B-roll'
        requested=f"{prefix} supporting: {preview}"
        primary=subject
    # Fine attributes and production concepts from a broad parent should not become
    # hard requirements on each child. Keep only an attribute literally present in
    # the child's narration.
    nl=narration.casefold()
    attrs=[a for a in parent.required_attributes or [] if str(a).casefold() in nl]
    intent=parent.visual_intent
    if intent=='comparison' and len(entities)<2:
        intent='literal_product' if entities else 'conceptual'
    specificity=parent.specificity if entities else 'normal'
    mode=_infer_match_mode(parent.match_mode,intent,narration,entities,specificity)
    return replace(parent,
        narration_text=narration,
        requested_visual=_semanticize_requested_visual(requested,narration),
        primary_subject=primary,
        required_entities=entities,
        required_attributes=attrs,
        visual_intent=intent,
        specificity=specificity,
        match_mode=mode)

def _set_timeline(reqs,audio_duration):
    for i,r in enumerate(reqs):
        r.timeline_start=0.0 if i==0 else (reqs[i-1].speech_end+r.speech_start)/2
        r.timeline_end=audio_duration if i==len(reqs)-1 else (r.speech_end+reqs[i+1].speech_start)/2

def _merge_pair(a,b):
    # Preserve the stronger/longer visual idea while keeping all narration words and
    # retrieval hints. This is only used when a visual unit would otherwise be too
    # short to support the hard minimum cut duration.
    da=max(0.0,a.speech_end-a.speech_start); db=max(0.0,b.speech_end-b.speech_start)
    anchor=a if da>=db else b
    spec_rank={'conceptual':0,'normal':1,'high':2,'strict':3}
    specificity=max((a.specificity,b.specificity),key=lambda x:spec_rank.get(x,1))
    mode_rank={'brand_filler':0,'thematic':1,'literal':2}
    match_mode=max((a.match_mode,b.match_mode),key=lambda x:mode_rank.get(x,1))
    merged=replace(anchor,
        start_word_id=a.start_word_id,
        end_word_id=b.end_word_id,
        narration_text=(a.narration_text+' '+b.narration_text).strip(),
        required_entities=_uniq([*a.required_entities,*b.required_entities]),
        required_attributes=_uniq([*a.required_attributes,*b.required_attributes]),
        preferred_shot_types=_uniq([*a.preferred_shot_types,*b.preferred_shot_types]),
        preferred_content_types=_uniq([*a.preferred_content_types,*b.preferred_content_types]),
        visual_concepts=_uniq([*a.visual_concepts,*b.visual_concepts]),
        avoid=_uniq([*a.avoid,*b.avoid]),
        specificity=specificity,
        match_mode=match_mode,
        speech_start=a.speech_start,
        speech_end=b.speech_end,
        timeline_start=0.0,
        timeline_end=0.0)
    return merged

def _merge_short_visual_units(reqs,audio_duration,min_seconds):
    if min_seconds<=0: return reqs
    _set_timeline(reqs,audio_duration)
    while len(reqs)>1:
        shorts=[i for i,r in enumerate(reqs) if r.duration < min_seconds-0.03]
        if not shorts: break
        i=min(shorts,key=lambda x:reqs[x].duration)
        if i==0:
            left,right=0,1
        elif i==len(reqs)-1:
            left,right=i-1,i
        else:
            # Merge into the neighbor that already owns the longer visual span. This
            # minimizes visual churn and preserves a stable, longer B-roll beat.
            prev_d=reqs[i-1].duration; next_d=reqs[i+1].duration
            if prev_d>=next_d: left,right=i-1,i
            else: left,right=i,i+1
        reqs=reqs[:left]+[_merge_pair(reqs[left],reqs[right])]+reqs[right+1:]
        _set_timeline(reqs,audio_duration)
    for n,r in enumerate(reqs,1): r.segment_id=f'vs_{n:04d}'
    return reqs



def _split_long_visual_units(reqs, words, audio_duration, target_seconds, max_seconds, min_seconds):
    """Deterministically split oversized semantic ideas into edit-paced visual beats.

    Qwen decides *what* each semantic unit means. This pass decides only how often
    the viewer should see a fresh piece of B-roll. Long-form faceless edits need
    regular visual refreshes even when the narration stays on one idea for 30-90s.
    The semantic requirement is copied to each slice, while narration/word bounds
    are exact and contiguous.
    """
    target=max(float(min_seconds), float(target_seconds))
    maximum=max(target, float(max_seconds))
    pos={w['id']:i for i,w in enumerate(words)}

    def split_once(r, parts):
        a,b=pos[r.start_word_id],pos[r.end_word_id]
        count=b-a+1
        parts=max(1,min(int(parts),count))
        if parts<=1:
            return [r]
        # Choose boundaries nearest equal speech-time targets, with at least one
        # word left for each remaining slice.
        cuts=[]; last=a-1
        speech_span=max(0.001,r.speech_end-r.speech_start)
        for k in range(1,parts):
            ideal=r.speech_start + speech_span*(k/parts)
            lo=last+1
            hi=b-(parts-k)
            if lo>hi: break
            idx=min(range(lo,hi+1), key=lambda i:abs(float(words[i]['end'])-ideal))
            cuts.append(idx); last=idx
        bounds=[]; start=a
        for end in [*cuts,b]:
            if end<start: continue
            chunk_words=words[start:end+1]
            x=_derive_split_child(r,chunk_words)
            x=replace(x,
                start_word_id=chunk_words[0]['id'],
                end_word_id=chunk_words[-1]['id'],
                speech_start=float(chunk_words[0]['start']),
                speech_end=float(chunk_words[-1]['end']),
                timeline_start=0.0,timeline_end=0.0)
            bounds.append(x); start=end+1
        return bounds or [r]

    # Two passes are enough in practice; repeat a few times in case long pauses
    # make midpoint-derived timeline spans larger than raw speech spans.
    out=list(reqs)
    for _ in range(4):
        _set_timeline(out,audio_duration)
        changed=False; nxt=[]
        for r in out:
            if r.duration<=maximum+0.03:
                nxt.append(r); continue
            parts=max(2, int(__import__('math').ceil(r.duration/target)))
            pieces=split_once(r,parts)
            if len(pieces)>1: changed=True
            nxt.extend(pieces)
        out=nxt
        if not changed: break
    _set_timeline(out,audio_duration)
    # Any tiny slice created around a pause is merged back; the next loop can then
    # split an oversized merged result again if necessary.
    out=_merge_short_visual_units(out,audio_duration,min_seconds)
    for _ in range(3):
        _set_timeline(out,audio_duration)
        if not any(r.duration>maximum+0.03 for r in out): break
        nxt=[]
        for r in out:
            if r.duration>maximum+0.03:
                parts=max(2,int(__import__('math').ceil(r.duration/target)))
                nxt.extend(split_once(r,parts))
            else: nxt.append(r)
        out=_merge_short_visual_units(nxt,audio_duration,min_seconds)
    _set_timeline(out,audio_duration)
    for n,r in enumerate(out,1): r.segment_id=f'vs_{n:04d}'
    return out


LONGFORM_CHUNK_TRIGGER_WORDS=320
LONGFORM_CHUNK_TARGET_WORDS=190
LONGFORM_REPAIR_EDGE_WORDS=12
LONGFORM_REPAIR_BOUNDARY_WORDS=18


def _build_planner_prompt(script_text, words):
    return (
    "You are the editorial planning stage of a LOCAL faceless rough-cut system that can only select REAL EXISTING indexed footage. "
    "Group the narration into meaningful VISUAL units, not blindly one sentence each. Every script word must be covered exactly once, in order, by contiguous start_word_id/end_word_id ranges. "
    "Keep a single visual idea together; split when what the viewer should see materially changes. If one sentence contrasts or moves between two distinct named products/entities, split at the natural clause/phrase boundary when possible. If a clean split is awkward, it is also valid to keep one comparison segment with both exact names in required_entities; the matcher can cover that segment with multiple sequential clips. "
    "Describe WHAT MUST BE VISIBLE, never how a hypothetical editor should film or manufacture it. requested_visual must name subjects, products, actions, components, or visible properties only. Do NOT put close-up, macro, wide, static, pan, tracking, zoom, X-ray, animation, split-screen, montage, sequential cuts, transitions, or motion graphics into requested_visual or required_attributes. Put framing preferences only in preferred_shot_types. "
    "For mechanical internals, request a real visible mechanical movement/gears/components rather than an X-ray or animation. "
    "required_attributes are SOFT visual hints only; they help search but do not need exact coverage. Put color, texture, markers, finishing, component detail and similar specifics there rather than making them hard requirements. "
    "The project forbids human faces. Add 'human face visible' to avoid and never request a face. Hands/wrists/bodies without a visible face are allowed. "
    "Assign match_mode to every semantic unit. Use literal SPARINGLY: only when an exact visible model/product/detail/process/comparison materially helps and the library could reasonably contain it. A brand mention alone does NOT make a beat literal. Use thematic when related topic/brand footage is enough. Use brand_filler for opinion, value judgment, status, community commentary, abstract claims, transitions, or narration that cannot sensibly be illustrated; same-brand/same-topic beauty B-roll is intentionally correct for those beats. Most long-form commentary beats should be thematic or brand_filler rather than literal. "
    "Prefer literal product/detail/process footage when match_mode is literal, but ordinary related B-roll is acceptable when perfect footage is unavailable. For named products/entities, put the exact name in required_entities and use specificity strict/high only when the exact subject actually matters. Exact named entities matter much more than fine visual attributes. Do not invent facts not stated in the script. "
    "Aim for long-form faceless-video pacing: a visual idea should normally refresh every 6-10 seconds. Do not create tiny visual units under 3 seconds, and avoid planning units longer than about 10 seconds. A deterministic pacing pass will split any oversized semantic idea while preserving word coverage. "
    "Do not invent graphics, overlays, presenters, community-consensus visuals, certification text, timelines, charts, or data graphics unless the narration explicitly requires a real visible object and the footage library could reasonably contain it. When the narration is abstract, request ordinary topical B-roll instead. "
    "preferred_shot_types and preferred_content_types are SOFT hints. Prefer the supported taxonomy when possible; do not invent a new hard requirement just to label a comparison or beauty shot. "
    "A requested_visual should be achievable by ordinary source footage from a manually collected library.\n\nSCRIPT:\n"+script_text+"\n\nWORD IDS:\n"+_word_text(words))


def _planner_range_anomaly(segs, ids, pos, edge_words=LONGFORM_REPAIR_EDGE_WORDS, boundary_words=LONGFORM_REPAIR_BOUNDARY_WORDS):
    """Return a reason when planner coverage is too broken for boundary stitching.

    v1.0.17 intentionally repairs tiny gaps/overlaps. It must never stretch one
    semantic segment across hundreds of omitted words, because that silently turns
    a partial model response into a fake full-script plan.
    """
    if not segs:
        return 'empty planner response'
    starts=[pos[s['start_word_id']] for s in segs]
    ends=[pos[s['end_word_id']] for s in segs]
    if starts[0] > int(edge_words):
        return f'planner omitted {starts[0]} leading words'
    trailing=(len(ids)-1)-ends[-1]
    if trailing > int(edge_words):
        return f'planner omitted {trailing} trailing words'
    for i in range(len(segs)):
        if ends[i] < starts[i]:
            return f'planner segment {i+1} is inverted'
        if i:
            delta=starts[i]-(ends[i-1]+1)
            if abs(delta) > int(boundary_words):
                kind='gap' if delta>0 else 'overlap'
                return f'planner has {abs(delta)}-word {kind} before segment {i+1}'
    return ''


def _chunk_word_ranges(words, target=LONGFORM_CHUNK_TARGET_WORDS):
    """Split long narration into sentence-aware contiguous planner chunks."""
    n=len(words)
    if not n:
        return []
    target=max(80,int(target)); out=[]; start=0
    while start<n:
        if n-start <= target+45:
            out.append((start,n-1)); break
        ideal=min(n-1,start+target-1)
        lo=max(start+80,ideal-35); hi=min(n-2,ideal+35)
        punct=[]
        for i in range(lo,hi+1):
            text=str(words[i].get('text','')).rstrip()
            if text.endswith(('.', '!', '?')):
                punct.append(i)
        end=min(punct,key=lambda i:abs(i-ideal)) if punct else ideal
        out.append((start,end)); start=end+1
    return out


def _fallback_chunk_segment(chunk_words, reason=''):
    """Last-resort safe editorial filler for a small chunk.

    A rough cut should continue with broad topical B-roll rather than fail because
    the planning model returned malformed JSON/ranges for one small chunk.
    """
    a,b=chunk_words[0]['id'],chunk_words[-1]['id']
    return {'start_word_id':a,'end_word_id':b,'visual_intent':'conceptual',
            'requested_visual':'Relevant real topical B-roll supporting this narration',
            'primary_subject':'','required_entities':[],'required_attributes':[],
            'preferred_shot_types':[],'preferred_content_types':[],
            'visual_concepts':[],'avoid':['human face visible'],'specificity':'conceptual',
            'match_mode':'brand_filler'}


def _plan_chunk(cfg, chunk_words, depth=0):
    ids=[w['id'] for w in chunk_words]; pos={x:i for i,x in enumerate(ids)}
    text=' '.join(str(w.get('text','')) for w in chunk_words).strip()
    planner_tokens=max(3072,int(cfg.qwen_num_predict))
    try:
        data=chat(cfg.ollama_url,cfg.reasoning_model,[{'role':'user','content':_build_planner_prompt(text,chunk_words)}],SCHEMA,planner_tokens,cfg.qwen_temperature)
        segs=data.get('segments',[])
        if not segs:
            raise ValueError('Planner returned no segments')
        segs=_repair_invalid_word_ids(segs,ids,pos)
        anomaly=_planner_range_anomaly(segs,ids,pos)
        if anomaly:
            raise ValueError(anomaly)
        return _repair_planner_ranges(segs,ids,pos)
    except Exception as exc:
        # For a malformed long chunk, divide and retry so one partial model response
        # cannot corrupt hundreds of narration words. Small chunks fall back to
        # intentional topical filler; no words are ever dropped.
        if len(chunk_words)>95 and depth<3:
            ranges=_chunk_word_ranges(chunk_words,max(80,len(chunk_words)//2))
            if len(ranges)>1:
                log.warning('Planner chunk %s..%s invalid (%s); splitting into %d smaller chunks',ids[0],ids[-1],exc,len(ranges))
                out=[]
                for a,b in ranges:
                    out.extend(_plan_chunk(cfg,chunk_words[a:b+1],depth+1))
                return out
        log.warning('Planner chunk %s..%s invalid (%s); using safe brand/topic filler fallback',ids[0],ids[-1],exc)
        return [_fallback_chunk_segment(chunk_words,str(exc))]


def _plan_raw_segments(cfg, script_text, words):
    if len(words) < LONGFORM_CHUNK_TRIGGER_WORDS:
        planner_tokens=max(4096,int(cfg.qwen_num_predict))
        data=chat(cfg.ollama_url,cfg.reasoning_model,[{'role':'user','content':_build_planner_prompt(script_text,words)}],SCHEMA,planner_tokens,cfg.qwen_temperature)
        segs=data.get('segments',[]); ids=[w['id'] for w in words]; pos={x:i for i,x in enumerate(ids)}
        if not segs: raise ValueError('Planner returned no segments')
        segs=_repair_invalid_word_ids(segs,ids,pos)
        anomaly=_planner_range_anomaly(segs,ids,pos)
        if anomaly:
            log.warning('Planner full response unsafe for stitching (%s); switching to chunked planning',anomaly)
            out=[]
            for a,b in _chunk_word_ranges(words): out.extend(_plan_chunk(cfg,words[a:b+1]))
            return out
        return _repair_planner_ranges(segs,ids,pos)
    ranges=_chunk_word_ranges(words)
    log.info('Long-form planner: %d aligned words -> %d focused chunks',len(words),len(ranges))
    out=[]
    for i,(a,b) in enumerate(ranges,1):
        log.info('Planning chunk %d/%d: %s..%s',i,len(ranges),words[a]['id'],words[b]['id'])
        out.extend(_plan_chunk(cfg,words[a:b+1]))
    return out

# v1.0.19: long-form planning no longer asks Qwen to invent word ranges or a
# verbose VisualRequirement object. Cuts are deterministic; Qwen only makes the
# small editorial decisions that benefit from reasoning. This dramatically
# reduces malformed/truncated JSON while preserving Editorial Freedom.
COMPACT_BATCH_SIZE=8
COMPACT_SCHEMA={"type":"object","properties":{"beats":{"type":"array","items":{"type":"object","properties":{
    "beat_id":{"type":"string"},
    "mode":{"type":"string"},
    "anchor":{"type":"string"},
    "visual":{"type":"string"},
    "entities":{"type":"array","items":{"type":"string"}},
},"required":["beat_id","mode","visual"],"additionalProperties":False}}},"required":["beats"],"additionalProperties":False}


def _paced_word_ranges(words, target_seconds=8.0, max_seconds=10.0, min_seconds=3.0):
    """Create stable sentence-aware edit beats without using an LLM.

    The narration alignment already tells us where cuts can happen. We target an
    ~8 second refresh, prefer nearby sentence/clause endings, and never ask Qwen
    to reproduce fragile word ids. The final tiny remainder is merged backward.
    """
    if not words:
        return []
    target=max(float(min_seconds),float(target_seconds))
    maximum=max(target,float(max_seconds))
    minimum=max(0.5,float(min_seconds))
    n=len(words); out=[]; start=0
    while start<n:
        start_t=float(words[start]['start'])
        remaining=float(words[-1]['end'])-start_t
        if remaining<=maximum+0.05:
            out.append((start,n-1)); break
        valid=[]
        for i in range(start,n):
            d=float(words[i]['end'])-start_t
            if d>=minimum-0.05 and d<=maximum+0.05:
                valid.append(i)
            if d>maximum+0.05:
                break
        if not valid:
            # Extremely sparse/odd timestamps: still make forward progress.
            end=min(n-1,start+1)
        else:
            punct=[i for i in valid if str(words[i].get('text','')).rstrip().endswith(('.', '!', '?', ';', ':'))]
            pool=punct or valid
            end=min(pool,key=lambda i:abs((float(words[i]['end'])-start_t)-target))
            # Avoid stranding a sub-minimum tail when moving the boundary a few
            # words earlier can preserve two legal beats.
            tail=float(words[-1]['end'])-float(words[end+1]['start']) if end+1<n else 0.0
            if 0.0<tail<minimum and len(out)>=0:
                alternatives=[i for i in valid if i<end and (float(words[-1]['end'])-float(words[i+1]['start']))>=minimum]
                if alternatives:
                    end=max(alternatives)
        out.append((start,end)); start=end+1
    if len(out)>1:
        a,b=out[-1]
        dur=float(words[b]['end'])-float(words[a]['start'])
        if dur<minimum-0.05:
            out[-2]=(out[-2][0],b); out.pop()
    return out


def _compact_mode(raw, narration='', entities=None):
    key=_norm_label(raw)
    aliases={
        'literal':'literal','exact':'literal','exact_match':'literal','specific':'literal',
        'thematic':'thematic','theme':'thematic','topic':'thematic','related':'thematic','topical':'thematic',
        'brand_filler':'brand_filler','brandfiller':'brand_filler','filler':'brand_filler',
        'abstract':'brand_filler','generic_filler':'brand_filler','topic_filler':'brand_filler',
    }
    if key in aliases:
        return aliases[key]
    # Unknown labels are soft. Reuse the conservative guardrails rather than
    # rejecting an otherwise useful compact response.
    return _infer_match_mode(None,'conceptual',narration,entities or [],'normal')


def _compact_prompt(beats):
    rows=[]
    for b in beats:
        rows.append(f"{b['beat_id']} | {b['narration']}")
    return (
        "You are planning B-roll for a faceless commentary video. The edit cuts are already fixed. "
        "For EACH beat below, return only a compact editorial classification. Do not create timestamps or word ranges. "
        "mode must be one of literal, thematic, brand_filler. Use literal sparingly: only when an exact visible product, model, component, process, or direct comparison materially helps. "
        "Use thematic when same-topic or same-brand footage is enough. Use brand_filler for opinion, status, community commentary, abstract claims, transitions, CTA, or narration that cannot sensibly be illustrated. "
        "anchor is the stable brand/product/topic the filler should stay on; never use CTA fragments such as 'tell me', 'comments', 'like', 'luxury counter', or 'next brand' as anchors. "
        "visual must be a short achievable description of REAL source footage, not graphics, overlays, split screens, animations, or invented shots. "
        "entities is optional and should contain at most two exact named products/brands that actually appear in that beat. "
        "Return exactly the same beat_id values. Keep each answer short.\n\nBEATS:\n" + '\n'.join(rows)
    )


def _compact_fallback_result(beat):
    return {'beat_id':beat['beat_id'],'mode':'brand_filler','anchor':'',
            'visual':'Relevant attractive real same-brand or same-topic B-roll','entities':[]}


def _compact_plan_batch(cfg, beats, depth=0):
    """Classify a small batch; split only the batch on malformed model output."""
    try:
        # The compact response is intentionally small. A bounded budget prevents
        # Qwen thinking/output from expanding into the failure mode seen in v1.0.18.
        tokens=max(768,min(2048,int(cfg.qwen_num_predict)))
        data=chat(cfg.ollama_url,cfg.reasoning_model,[{'role':'user','content':_compact_prompt(beats)}],COMPACT_SCHEMA,tokens,cfg.qwen_temperature)
        raw=data.get('beats',[]) or []
        by_id={str(x.get('beat_id','')).strip():x for x in raw if isinstance(x,dict)}
        out=[]
        for beat in beats:
            x=by_id.get(beat['beat_id'])
            if x is None:
                out.append(_compact_fallback_result(beat)); continue
            ents=[_clean_entity(e) for e in (x.get('entities') or []) if _clean_entity(e)][:2]
            anchor=_clean_entity(x.get('anchor',''))
            if _is_generic_context(anchor): anchor=''
            visual=' '.join(str(x.get('visual') or '').split()).strip()
            if not visual: visual='Relevant attractive real same-brand or same-topic B-roll'
            out.append({'beat_id':beat['beat_id'],
                        'mode':_compact_mode(x.get('mode'),beat['narration'],ents),
                        'anchor':anchor,'visual':visual,'entities':ents})
        return out
    except Exception as exc:
        if len(beats)>1 and depth<3:
            mid=len(beats)//2
            log.warning('Compact planner batch %s..%s invalid (%s); splitting into 2 smaller batches',beats[0]['beat_id'],beats[-1]['beat_id'],exc)
            return _compact_plan_batch(cfg,beats[:mid],depth+1)+_compact_plan_batch(cfg,beats[mid:],depth+1)
        log.warning('Compact planner beat %s invalid (%s); using safe brand/topic filler fallback',beats[0]['beat_id'],exc)
        return [_compact_fallback_result(beats[0])]


def _compact_intent(mode, narration, entities):
    nl=str(narration or '').casefold()
    if mode=='literal':
        if len(entities)>=2: return 'comparison'
        if any(c in nl for c in _DETAIL_CUES): return 'literal_product_detail'
        return 'literal_product'
    return 'conceptual'


def _plan_compact_longform(cfg, words, audio_duration):
    ranges=_paced_word_ranges(
        words,
        float(getattr(cfg,'target_visual_segment_seconds',8.0)),
        float(getattr(cfg,'max_visual_segment_seconds',10.0)),
        float(getattr(cfg,'min_visual_clip_seconds',3.0)))
    beats=[]
    for i,(a,b) in enumerate(ranges,1):
        narration=' '.join(str(w.get('text','')) for w in words[a:b+1]).strip()
        beats.append({'beat_id':f'b{i:04d}','a':a,'b':b,'narration':narration})
    log.info('Compact long-form planner: %d aligned words -> %d fixed edit beats -> %d Qwen batch(es)',
             len(words),len(beats),(len(beats)+COMPACT_BATCH_SIZE-1)//COMPACT_BATCH_SIZE)
    decisions=[]
    for i in range(0,len(beats),COMPACT_BATCH_SIZE):
        batch=beats[i:i+COMPACT_BATCH_SIZE]
        log.info('Classifying compact planner batch %d/%d: %s..%s',
                 i//COMPACT_BATCH_SIZE+1,(len(beats)+COMPACT_BATCH_SIZE-1)//COMPACT_BATCH_SIZE,
                 batch[0]['beat_id'],batch[-1]['beat_id'])
        decisions.extend(_compact_plan_batch(cfg,batch))
    by_id={x['beat_id']:x for x in decisions}
    reqs=[]
    for n,beat in enumerate(beats,1):
        d=by_id.get(beat['beat_id'],_compact_fallback_result(beat))
        a,b=beat['a'],beat['b']; narration=beat['narration']
        entities=[e for e in d.get('entities',[]) if not _is_generic_context(e)]
        anchor=_clean_entity(d.get('anchor',''))
        if anchor and not _is_generic_context(anchor) and anchor.casefold() not in {e.casefold() for e in entities}:
            entities=[anchor,*entities][:2]
        mode=_compact_mode(d.get('mode'),narration,entities)
        intent=_compact_intent(mode,narration,entities)
        specificity='high' if mode=='literal' and entities else ('conceptual' if mode=='brand_filler' else 'normal')
        primary=anchor or (entities[0] if entities else '')
        visual=_semanticize_requested_visual(d.get('visual',''),narration)
        r=VisualRequirement(
            segment_id=f'vs_{n:04d}',start_word_id=words[a]['id'],end_word_id=words[b]['id'],
            narration_text=narration,visual_intent=intent,requested_visual=visual,
            primary_subject=primary,required_entities=entities,required_attributes=[],
            preferred_shot_types=[],preferred_content_types=[],visual_concepts=[],
            avoid=['human face visible'],specificity=specificity,match_mode=mode,
            speech_start=float(words[a]['start']),speech_end=float(words[b]['end']))
        reqs.append(r)
    _set_timeline(reqs,audio_duration)
    # Context is assigned globally after every beat has been classified, so a safe
    # fallback near the beginning can still inherit the dominant brand found later.
    reqs=_assign_context_anchors(reqs)
    return reqs


def plan_segments(cfg,script_text,words,audio_duration):
    # Long-form commentary uses the compact v1.0.19 architecture: deterministic
    # edit beats + tiny Qwen classifications. Short scripts retain the legacy
    # semantic planner where the old schema is already reliable.
    if len(words) >= LONGFORM_CHUNK_TRIGGER_WORDS:
        return _plan_compact_longform(cfg,words,audio_duration)
    segs=_plan_raw_segments(cfg,script_text,words)
    ids=[w['id'] for w in words]; pos={x:i for i,x in enumerate(ids)}
    # Chunks are independently contiguous and together span the full word list.
    # A final tiny stitch handles sentence-boundary rounding only.
    segs=_repair_planner_ranges(segs,ids,pos)
    reqs=[]; expected=0
    for n,s in enumerate(segs,1):
        a,b=pos[s['start_word_id']],pos[s['end_word_id']]
        if a!=expected or b<a: raise ValueError('Planner ranges are not contiguous/ordered')
        text=' '.join(w['text'] for w in words[a:b+1])
        requested=_semanticize_requested_visual(s['requested_visual'],text)
        attrs=_filter_required_attributes(s['required_attributes'],text)
        mode=_infer_match_mode(s.get('match_mode'),s['visual_intent'],text,s['required_entities'],s['specificity'])
        shot_types=_normalize_shot_types(s.get('preferred_shot_types',[]))
        content_types=_normalize_content_types(s.get('preferred_content_types',[]))
        r=VisualRequirement(f'vs_{n:04d}',s['start_word_id'],s['end_word_id'],text,s['visual_intent'],requested,s['primary_subject'],s['required_entities'],attrs,shot_types,content_types,s['visual_concepts'],list(dict.fromkeys([*s['avoid'],'human face visible'])),s['specificity'],match_mode=mode)
        r.speech_start=words[a]['start']; r.speech_end=words[b]['end']; reqs.append(r); expected=b+1
    if expected!=len(words): raise ValueError('Planner did not cover all script words')
    # Deterministic pacing for long-form faceless edits. Qwen owns semantics; this
    # pass guarantees that one broad idea cannot occupy 30-90 seconds of timeline.
    mn=float(getattr(cfg,'min_visual_clip_seconds',3.0))
    reqs=_merge_short_visual_units(reqs,audio_duration,mn)
    reqs=_split_long_visual_units(
        reqs,words,audio_duration,
        float(getattr(cfg,'target_visual_segment_seconds',8.0)),
        float(getattr(cfg,'max_visual_segment_seconds',10.0)),mn)
    reqs=_assign_context_anchors(reqs)
    return reqs
