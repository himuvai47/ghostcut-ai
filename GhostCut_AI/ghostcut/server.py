from __future__ import annotations
import threading, webbrowser
import uvicorn
from pathlib import Path
from .settings import SettingsStore

def main():
    root=Path(__file__).resolve().parents[1]; s=SettingsStore(root).value
    url=f"http://{s.host}:{s.port}"
    if s.auto_open_browser: threading.Timer(1.0,lambda:webbrowser.open(url)).start()
    uvicorn.run("ghostcut.app:app",host=s.host,port=s.port,reload=False,log_level="warning")
if __name__=="__main__": main()
