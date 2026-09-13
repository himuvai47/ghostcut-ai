from __future__ import annotations
import os, subprocess


def _powershell_dialog(script: str) -> str:
    p=subprocess.run(["powershell.exe","-NoProfile","-STA","-ExecutionPolicy","Bypass","-Command",script],capture_output=True,text=True,encoding="utf-8",errors="replace")
    if p.returncode!=0:
        raise RuntimeError((p.stderr or p.stdout or "PowerShell dialog failed").strip())
    return p.stdout.strip().splitlines()[-1].strip() if p.stdout.strip() else ""


def choose_file(title: str, kinds: str = "all") -> str:
    if os.name=="nt":
        filt={
            "script":"Text files (*.txt)|*.txt|All files (*.*)|*.*",
            "audio":"Audio files (*.mp3;*.wav;*.m4a;*.aac;*.flac)|*.mp3;*.wav;*.m4a;*.aac;*.flac|All files (*.*)|*.*",
            "all":"All files (*.*)|*.*",
        }.get(kinds,"All files (*.*)|*.*")
        title=title.replace("'","''"); filt=filt.replace("'","''")
        return _powershell_dialog(f"Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.OpenFileDialog; $d.Title='{title}'; $d.Filter='{filt}'; if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){{[Console]::WriteLine($d.FileName)}}")
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    types={"script":[("Text files","*.txt"),("All files","*.*")],"audio":[("Audio files","*.mp3 *.wav *.m4a *.aac *.flac"),("All files","*.*")],"all":[("All files","*.*")]}
    try:return filedialog.askopenfilename(title=title,filetypes=types.get(kinds,types["all"])) or ""
    finally:root.destroy()


def choose_folder(title: str) -> str:
    if os.name=="nt":
        title=title.replace("'","''")
        return _powershell_dialog(f"Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='{title}'; $d.ShowNewFolderButton=$true; if($d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK){{[Console]::WriteLine($d.SelectedPath)}}")
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk(); root.withdraw()
    try:return filedialog.askdirectory(title=title,mustexist=True) or ""
    finally:root.destroy()
