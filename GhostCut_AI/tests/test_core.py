from __future__ import annotations
import json, sqlite3, tempfile
from pathlib import Path
from ghostcut.libraries import face_metadata_status, library_info
from ghostcut.projects import ProjectStore
from ghostcut.settings import GhostCutSettings, SettingsStore
from ghostcut.utils import slugify


def mk_settings(root: Path) -> GhostCutSettings:
    return GhostCutSettings(
        app_root=str(root/'app'),facelessrc_root=str(root),agent1_root=str(root/'Agent1_Indexing'),agent2_root=str(root/'Agent2_VideoCreation'),agent3_root=str(root/'Agent3_Renderer'),projects_root=str(root/'projects'),indexes_root=str(root/'indexes')
    )


def make_face_db(ws: Path, version=2):
    ws.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(ws/'footage_index.db')
    con.execute('CREATE TABLE clips (clip_id TEXT, face_status TEXT, face_scan_version INTEGER)')
    con.executemany('INSERT INTO clips VALUES (?,?,?)', [('a','no_face',version),('b','face_visible',version),('c','uncertain',version)])
    con.execute('CREATE TABLE sources (source_id TEXT)'); con.execute("INSERT INTO sources VALUES ('s1')")
    con.commit(); con.close()
    (ws/'qc_report.json').write_text('{"passed": true}')


def test_slugify():
    assert slugify('  Mechanical Watch Story!  ') == 'mechanical-watch-story'
    assert slugify('***') == 'ghostcut-project'


def test_face_metadata_ready(tmp_path):
    ws=tmp_path/'idx'; make_face_db(ws,2)
    st=face_metadata_status(ws)
    assert st['ready'] is True
    assert st['counts']['no_face']==1
    assert st['counts']['face_visible']==1


def test_face_metadata_stale(tmp_path):
    ws=tmp_path/'idx'; make_face_db(ws,1)
    st=face_metadata_status(ws)
    assert st['ready'] is False


def test_face_metadata_missing_columns(tmp_path):
    ws=tmp_path/'idx'; ws.mkdir(); con=sqlite3.connect(ws/'footage_index.db'); con.execute('CREATE TABLE clips (clip_id TEXT)'); con.close()
    st=face_metadata_status(ws)
    assert st['ready'] is False
    assert 'missing' in st['reason']


def test_library_info(tmp_path):
    ws=tmp_path/'idx'; make_face_db(ws,2)
    x=library_info(ws)
    assert x['clip_count']==3 and x['source_count']==1 and x['qc_passed'] is True


def test_project_store_copies_inputs(tmp_path):
    s=mk_settings(tmp_path); ps=ProjectStore(s)
    script=tmp_path/'script.txt'; script.write_text('hello')
    audio=tmp_path/'voice.mp3'; audio.write_bytes(b'audio')
    idx=tmp_path/'idx'; make_face_db(idx,2)
    p=ps.create('My Test',str(script),str(audio),'existing',str(idx),'')
    assert Path(p.script_path).read_text()=='hello'
    assert Path(p.audio_path).read_bytes()==b'audio'
    assert Path(p.root,'project.json').exists()
    assert ps.get(p.id).name=='My Test'


def test_new_project_assigns_index_workspace(tmp_path):
    s=mk_settings(tmp_path); ps=ProjectStore(s)
    script=tmp_path/'script.txt'; script.write_text('hello')
    audio=tmp_path/'voice.mp3'; audio.write_bytes(b'audio')
    footage=tmp_path/'footage'; footage.mkdir()
    p=ps.create('New Library',str(script),str(audio),'new','',str(footage))
    assert Path(p.library_path).name=='new-library-library'


def test_settings_store_defaults_and_update(tmp_path):
    app=tmp_path/'GhostCut_AI'; (app/'data').mkdir(parents=True)
    ss=SettingsStore(app)
    assert Path(ss.value.facelessrc_root)==tmp_path
    ss.update({'ollama_url':'http://localhost:9999'})
    assert SettingsStore(app).value.ollama_url=='http://localhost:9999'


def test_v100_completed_project_normalizes_pipeline_state(tmp_path):
    s=mk_settings(tmp_path); ps=ProjectStore(s)
    root=Path(s.projects_root)/'legacy_123'; root.mkdir(parents=True)
    a2=root/'agent2_output'; a3=root/'agent3_output'; a2.mkdir(); a3.mkdir(); (a3/'rough_cut.mp4').write_bytes(b'x')
    legacy={
        'id':'gc_legacy','name':'Legacy','slug':'legacy','root':str(root),'script_path':str(tmp_path/'s.txt'),'audio_path':str(tmp_path/'a.mp3'),
        'library_mode':'existing','library_path':str(tmp_path/'idx'),'status':'completed','agent2_workspace':str(a2),'agent3_workspace':str(a3),'rough_cut_path':str(a3/'rough_cut.mp4')
    }
    (root/'project.json').write_text(json.dumps(legacy))
    p=ps.get('gc_legacy')
    assert p.stage_states['match']=='complete' and p.stage_states['render']=='complete' and p.stage_states['done']=='complete'
    assert p.pipeline_stage=='done' and p.pipeline_progress==100


def test_write_json_retries_transient_windows_permission_error(tmp_path, monkeypatch):
    import os
    from ghostcut.utils import write_json, read_json
    target=tmp_path/'project.json'
    real_replace=os.replace
    calls={'n':0}
    def flaky_replace(src,dst):
        calls['n']+=1
        if calls['n']<3:
            raise PermissionError(5,'Access is denied',str(dst))
        return real_replace(src,dst)
    monkeypatch.setattr('ghostcut.utils.os.replace',flaky_replace)
    write_json(target,{'ok':True},replace_retries=4)
    assert read_json(target)=={'ok':True}
    assert calls['n']==3
    assert not list(tmp_path.glob('.*.tmp'))


def test_write_json_uses_unique_temp_files_for_concurrent_writes(tmp_path):
    import threading
    from ghostcut.utils import write_json, read_json
    target=tmp_path/'project.json'
    errors=[]
    def worker(i):
        try:
            for n in range(20):
                write_json(target,{'worker':i,'n':n})
        except Exception as exc:
            errors.append(exc)
    threads=[threading.Thread(target=worker,args=(i,)) for i in range(4)]
    for t in threads:t.start()
    for t in threads:t.join()
    assert not errors
    data=read_json(target)
    assert data['worker'] in range(4)
    assert 0 <= data['n'] < 20
    assert not list(tmp_path.glob('.*.tmp'))
