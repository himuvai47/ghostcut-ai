from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from . import __version__
from .config import Agent3Config
from .pipeline import Agent3Pipeline

def _logger(ws:Path,verbose=False):
    ws.mkdir(parents=True,exist_ok=True); (ws/'logs').mkdir(exist_ok=True)
    log=logging.getLogger('agent3'); log.handlers.clear(); log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt=logging.Formatter('%(levelname)s | %(message)s')
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    from datetime import datetime
    fh=logging.FileHandler(ws/'logs'/f"agent3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",encoding='utf-8'); fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')); log.addHandler(fh)
    return log

def main(argv=None):
    ap=argparse.ArgumentParser(description='FacelessRC Agent 3 deterministic FFmpeg renderer')
    ap.add_argument('--timeline',required=True)
    ap.add_argument('--workspace',required=True)
    ap.add_argument('--config')
    ap.add_argument('--force',action='store_true')
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args(argv)
    ws=Path(args.workspace); log=_logger(ws,args.verbose)
    log.info(f'Agent 3 v{__version__} starting')
    log.info(f'Workspace: {ws.resolve()}')
    try:
        cfg=Agent3Config.load(Path(args.config) if args.config else None)
        summary=Agent3Pipeline(cfg,ws,log).run(Path(args.timeline),force=args.force)
        print('\n'+json.dumps(summary,indent=2))
        print(f"\nRough cut: {summary['output']}")
        print(f"QC:        {(ws/'qc_report.json').resolve()}")
        return 0
    except Exception as e:
        log.error(str(e)); return 1
