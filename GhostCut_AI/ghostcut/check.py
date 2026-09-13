from __future__ import annotations
import json, platform, sys
from pathlib import Path
from . import __version__
from .health import health
from .settings import SettingsStore

def main():
    root=Path(__file__).resolve().parents[1]
    h=health(SettingsStore(root).value)
    payload={"ghostcut_version":__version__,"python":platform.python_version(),**h}
    print(json.dumps(payload,indent=2))
    return 0 if h["passed"] else 2
if __name__=="__main__": raise SystemExit(main())
