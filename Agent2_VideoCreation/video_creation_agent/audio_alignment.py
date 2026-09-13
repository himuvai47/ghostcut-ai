from __future__ import annotations
import json, re, subprocess
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path
from .models import TimedWord
from .utils import normalize_word, script_words

SPECIAL_RE=re.compile(r'^\[_.*_\]$')

def _ffprobe_duration(path:Path)->float:
    p=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(path)],capture_output=True,text=True,check=True)
    return float(p.stdout.strip())

def prepare_wav(audio:Path,out:Path):
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(audio),'-ar','16000','-ac','1','-c:a','pcm_s16le',str(out)],check=True)

def run_whisper(audio:Path,workspace:Path,cfg,root:Path)->Path:
    wav=workspace/'cache'/'audio_16k.wav'; prepare_wav(audio,wav)
    cli=(root/cfg.whisper_cli).resolve() if not Path(cfg.whisper_cli).is_absolute() else Path(cfg.whisper_cli)
    model=(root/cfg.whisper_model_path).resolve() if not Path(cfg.whisper_model_path).is_absolute() else Path(cfg.whisper_model_path)
    if not cli.exists(): raise FileNotFoundError(f'whisper-cli not found: {cli}. Run SETUP_WHISPER_CPP.ps1')
    if not model.exists(): raise FileNotFoundError(f'Whisper model not found: {model}. Run SETUP_WHISPER_CPP.ps1')
    outbase=workspace/'cache'/'whisper'
    cmd=[str(cli),'-m',str(model),'-f',str(wav),'-l',cfg.whisper_language,'-ojf','-sow','-np','-of',str(outbase)]
    # Deliberately no --vad: current whisper.cpp token JSON can use the wrong timebase with VAD.
    subprocess.run(cmd,check=True)
    out=Path(str(outbase)+'.json')
    if not out.exists(): raise RuntimeError(f'whisper-cli did not create {out}')
    return out

def extract_asr_words(data:dict):
    words=[]
    for seg in data.get('transcription',[]):
        for tok in seg.get('tokens',[]):
            text=str(tok.get('text','')).strip()
            if not text or SPECIAL_RE.match(text): continue
            norms=re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*",text)
            if not norms: continue
            off=tok.get('offsets') or {}; st=float(off.get('from',0))/1000.0; en=float(off.get('to',off.get('from',0)))/1000.0
            if en < st: en=st
            span=max(0.001,en-st); n=len(norms)
            for i,w in enumerate(norms):
                words.append({'text':w,'norm':normalize_word(w),'start':st+span*i/n,'end':st+span*(i+1)/n})
    return words

def align_script(script_text:str, whisper_json:dict, audio_duration:float):
    sw=script_words(script_text); sn=[normalize_word(x) for x in sw]
    aw=extract_asr_words(whisper_json); an=[x['norm'] for x in aw]
    sm=SequenceMatcher(a=sn,b=an,autojunk=False)
    mapped={}
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='equal':
            for k in range(i2-i1): mapped[i1+k]=aw[j1+k]
        elif tag=='replace':
            m=min(i2-i1,j2-j1)
            for k in range(m):
                if SequenceMatcher(a=sn[i1+k],b=an[j1+k]).ratio()>=0.72: mapped[i1+k]=aw[j1+k]
    result=[]
    for i,w in enumerate(sw):
        if i in mapped:
            a=mapped[i]; result.append(TimedWord(f'w{i+1:05d}',w,a['start'],a['end'],True,a['text']))
        else:
            result.append(TimedWord(f'w{i+1:05d}',w,-1,-1,False,None))
    # interpolate unmatched runs between known words, preserving monotonic time
    known=[i for i,x in enumerate(result) if x.aligned]
    if not known:
        raise RuntimeError('Unable to align any script words to voiceover transcription')
    for i,x in enumerate(result):
        if x.aligned: continue
        prev=max((k for k in known if k<i),default=None); nxt=min((k for k in known if k>i),default=None)
        if prev is None:
            right=result[nxt].start; count=nxt+1; st=max(0,right-0.28*count); en=st+0.24
        elif nxt is None:
            left=result[prev].end; count=i-prev; st=min(audio_duration,left+0.20*count); en=min(audio_duration,st+0.18)
        else:
            left=result[prev].end; right=result[nxt].start; total=nxt-prev; pos=i-prev
            st=left+(right-left)*(pos-0.15)/total; en=left+(right-left)*(pos+0.65)/total
        x.start=max(0,min(audio_duration,st)); x.end=max(x.start,min(audio_duration,en))
    for i in range(1,len(result)):
        if result[i].start < result[i-1].start: result[i].start=result[i-1].start
        if result[i].end < result[i].start: result[i].end=result[i].start
    ratio=len(mapped)/max(1,len(sw))
    return result,ratio

def build_alignment(script_path:Path,audio_path:Path,workspace:Path,cfg,root:Path):
    duration=_ffprobe_duration(audio_path)
    out_json=run_whisper(audio_path,workspace,cfg,root)
    data=json.loads(out_json.read_text(encoding='utf-8'))
    words,ratio=align_script(script_path.read_text(encoding='utf-8'),data,duration)
    return {'audio_duration':duration,'alignment_ratio':round(ratio,4),'words':[asdict(w) for w in words]}
