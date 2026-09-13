from pathlib import Path

def test_ui_assets_exist():
    root=Path(__file__).resolve().parents[1]
    html=(root/'static'/'index.html').read_text()
    css=(root/'static'/'styles.css').read_text()
    js=(root/'static'/'app.js').read_text()
    assert 'GhostCut AI' in html
    assert 'CREATE ROUGH CUT' in html
    assert '--cyan:#50e6ff' in css
    assert '/api/projects' in js and '/api/libraries' in js
    assert 'View Pipeline' in js and 'run/from_agent2' not in js
    assert '/run/${mode}' in js
    assert "e.status===404" in js
    assert "Recovered stale pipeline state" in js

def test_powershell_entrypoints_exist():
    root=Path(__file__).resolve().parents[1]
    for name in ['SETUP_WINDOWS.ps1','CHECK_GHOSTCUT.ps1','RUN_GHOSTCUT.ps1','RUN_TESTS.ps1']:
        assert (root/name).exists()
