from __future__ import annotations
import argparse, json, sys, urllib.request
from pathlib import Path
from . import __version__
from .config import Agent2Config
from .pipeline import Agent2Pipeline
from .utils import setup_logger

def _check_model(base, model):
    try:
        with urllib.request.urlopen(base.rstrip('/')+'/api/tags',timeout=4) as r:
            data=json.loads(r.read().decode())
        names={m.get('name','') for m in data.get('models',[])}|{m.get('model','') for m in data.get('models',[])}
        return any(x==model or x.startswith(model+':') or model.startswith(x+':') for x in names)
    except Exception:
        return False

def precheck(cfg:Agent2Config,root:Path):
    import shutil
    out={'agent2_version':__version__,'python':sys.version.split()[0],
         'ffmpeg':bool(shutil.which('ffmpeg')),'ffprobe':bool(shutil.which('ffprobe')),
         'ollama_reasoning_model':_check_model(cfg.ollama_url,cfg.reasoning_model),
         'ollama_embedding_model':_check_model(cfg.ollama_url,cfg.embedding_model)}
    cli=(root/cfg.whisper_cli).resolve() if not Path(cfg.whisper_cli).is_absolute() else Path(cfg.whisper_cli)
    model=(root/cfg.whisper_model_path).resolve() if not Path(cfg.whisper_model_path).is_absolute() else Path(cfg.whisper_model_path)
    out['whisper_cli']=cli.exists(); out['whisper_model']=model.exists()
    if cli.exists():
        import subprocess
        try:
            cp=subprocess.run([str(cli),'-h'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
            out['whisper_cli_runnable']=cp.returncode in (0,1)
        except Exception:
            out['whisper_cli_runnable']=False
    else:
        out['whisper_cli_runnable']=False
    out['passed']=all(v for k,v in out.items() if k not in {'agent2_version','python'})
    return out

def main(argv=None):
    ap=argparse.ArgumentParser(description='FacelessRC Agent 2 - Video Creation Agent')
    ap.add_argument('--script',type=Path); ap.add_argument('--audio',type=Path); ap.add_argument('--agent1-workspace',type=Path); ap.add_argument('--workspace',type=Path)
    ap.add_argument('--config',type=Path); ap.add_argument('--force',action='store_true'); ap.add_argument('--skip-global-review',action='store_true'); ap.add_argument('--verbose',action='store_true'); ap.add_argument('--check',action='store_true')
    args=ap.parse_args(argv); root=Path(__file__).resolve().parents[1]; cfg=Agent2Config.load(args.config)
    if args.check:
        result=precheck(cfg,root); print(json.dumps(result,indent=2)); return 0 if result['passed'] else 2
    for name in ('script','audio','agent1_workspace','workspace'):
        if getattr(args,name) is None: ap.error(f'--{name.replace("_","-")} is required')
    for p,n in [(args.script,'script'),(args.audio,'audio')]:
        if not p.exists(): raise SystemExit(f'{n} not found: {p}')
    logger,log_path=setup_logger(args.workspace,args.verbose)
    logger.info(f'Agent 2 v{__version__} starting'); logger.info(f'Workspace: {args.workspace}')
    pipe=Agent2Pipeline(cfg,args.workspace,root,logger)
    try:
        summary=pipe.run(args.script,args.audio,args.agent1_workspace,args.force,args.skip_global_review)
        print('\n'+json.dumps(summary,indent=2)); print(f'\nTimeline: {args.workspace / "timeline.json"}\nQC:       {args.workspace / "qc_report.json"}\nLog:      {log_path}')
        return 0 if summary['qc_passed'] else 3
    except Exception:
        logger.exception('Agent 2 failed'); print(f'Log: {log_path}',file=sys.stderr); return 1
    finally: pipe.close()

if __name__=='__main__': raise SystemExit(main())
