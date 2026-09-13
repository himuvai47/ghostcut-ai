from __future__ import annotations
from .ollama import chat_fast
SCHEMA={"type":"object","properties":{"issues":{"type":"array","items":{"type":"object","properties":{"segment_id":{"type":"string"},"severity":{"type":"string","enum":["minor","major"]},"issue":{"type":"string"},"action":{"type":"string","enum":["keep","rematch","no_match"]}},"required":["segment_id","severity","issue","action"],"additionalProperties":False}}},"required":["issues"],"additionalProperties":False}

def review(cfg,summary):
    prompt=(
    "Review ONLY the supplied RISK SEGMENTS from a FACELESS rough-cut. These segments are here because a clip was repeated. "
    "Your primary job is to decide whether the repetition is obviously distracting or the repeated footage is completely unrelated. "
    "General relevance is enough; topical B-roll is enough. NEVER require a visible buyer, presenter, person, graphic, chart, overlay, certification text, community, store, or any other literal illustration of abstract narration. "
    "NEVER flag missing fine details such as color, hour markers, texture, finishing, framing, camera motion, or exact component action. "
    "Evaluate multi-entity segments COLLECTIVELY: separate sequential clips may cover different named products. Only treat a named product/entity as hard when that exact name appears in required_entities. "
    "Do not inspect or invent problems in segments that were not supplied. When in doubt, KEEP the footage. Return only genuine major failures.\n"+str(summary))
    return chat_fast(cfg.ollama_url,cfg.reasoning_model,[{'role':'user','content':prompt}],SCHEMA,min(int(cfg.qwen_num_predict),768),cfg.qwen_temperature).get('issues',[])
