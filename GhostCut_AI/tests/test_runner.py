from __future__ import annotations
import sqlite3
from pathlib import Path
from ghostcut.projects import ProjectStore
from ghostcut.runner import JobRegistry, PipelineRunner
from ghostcut.settings import GhostCutSettings


def settings(root: Path):
    for a,run in [('Agent1_Indexing','RUN_AGENT1.ps1'),('Agent2_VideoCreation','RUN_AGENT2.ps1'),('Agent3_Renderer','RUN_AGENT3.ps1')]:
        d=root/a; d.mkdir(); (d/run).write_text('# fake')
    return GhostCutSettings(app_root=str(root/'GhostCut_AI'),facelessrc_root=str(root),agent1_root=str(root/'Agent1_Indexing'),agent2_root=str(root/'Agent2_VideoCreation'),agent3_root=str(root/'Agent3_Renderer'),projects_root=str(root/'projects'),indexes_root=str(root/'indexes'))


def face_db(ws:Path):
    ws.mkdir(parents=True,exist_ok=True); con=sqlite3.connect(ws/'footage_index.db')
    con.execute('DROP TABLE IF EXISTS clips')
    con.execute('CREATE TABLE clips (clip_id TEXT, face_status TEXT, face_scan_version INTEGER)'); con.execute("INSERT INTO clips VALUES ('x','no_face',2)"); con.commit(); con.close(); (ws/'qc_report.json').write_text('{"passed":true}')

class FakeRunner(PipelineRunner):
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.calls=[]
    def _powershell(self,job,script,args,progress_cb=None):
        self.calls.append((script.name,list(args)))
        if script.name=='RUN_AGENT1.ps1':
            ws=Path(args[args.index('-Workspace')+1]); face_db(ws); self._log(job,'INFO | Agent 1 complete')
        elif script.name=='RUN_AGENT2.ps1':
            ws=Path(args[args.index('-Workspace')+1]); ws.mkdir(parents=True,exist_ok=True); (ws/'timeline.json').write_text('{"audio":{"source":"fake","duration":3},"segments":[]}')
            for line in ['INFO | Visual segments: 2','INFO | Matching vs_0001: A','INFO | Matching vs_0002: B']:
                self._log(job,line); progress_cb and progress_cb(line)
        elif script.name=='RUN_AGENT3.ps1':
            ws=Path(args[args.index('-Workspace')+1]); ws.mkdir(parents=True,exist_ok=True); (ws/'rough_cut.mp4').write_bytes(b'fake'); (ws/'qc_report.json').write_text('{"passed":true}')
            line='INFO | Rendering clip 2/2'; self._log(job,line); progress_cb and progress_cb(line)


def test_existing_library_skips_agent1(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Existing',str(script),str(audio),'existing',str(lib),''); job=reg.create(p.id); runner._run(p,job)
    assert job.status=='completed'
    assert [x[0] for x in runner.calls]==['RUN_AGENT2.ps1','RUN_AGENT3.ps1']
    assert Path(ps.get(p.id).rough_cut_path).exists()


def test_new_footage_runs_all_agents(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); footage=tmp_path/'footage'; footage.mkdir()
    p=ps.create('Fresh',str(script),str(audio),'new','',str(footage)); job=reg.create(p.id); runner._run(p,job)
    assert job.status=='completed'
    assert [x[0] for x in runner.calls]==['RUN_AGENT1.ps1','RUN_AGENT2.ps1','RUN_AGENT3.ps1']
    assert job.progress==100 and job.stage_states['done']=='complete'


def test_stale_existing_index_triggers_face_backfill(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'old'; lib.mkdir(parents=True); con=sqlite3.connect(lib/'footage_index.db'); con.execute('CREATE TABLE clips (clip_id TEXT)'); con.close(); (lib/'qc_report.json').write_text('{"passed":true}')
    p=ps.create('Old',str(script),str(audio),'existing',str(lib),''); job=reg.create(p.id); runner._run(p,job)
    assert job.status=='completed'
    assert runner.calls[0][0]=='RUN_AGENT1.ps1'
    assert '-FaceBackfill' in runner.calls[0][1]


def prepared_job(runner, reg, p, mode):
    states=runner._prepare_run(p,mode)
    return reg.create(p.id,mode,states,str(Path(p.root)/'logs'/'ghostcut_pipeline.log'))


def test_agent2_only_marks_render_stale_and_runs_only_agent2(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Stage Match',str(script),str(audio),'existing',str(lib),'')
    # Pretend the old project had a complete render.
    p.stage_states.update({'preflight':'complete','index':'skipped','face':'complete','match':'complete','render':'complete','done':'complete'}); p.status='completed'; ps.save(p)
    job=prepared_job(runner,reg,p,'agent2'); runner._run(p,job,'agent2')
    saved=ps.get(p.id)
    assert job.status=='completed' and saved.status=='stale'
    assert saved.stage_states['match']=='complete'
    assert saved.stage_states['render']=='stale' and saved.stage_states['done']=='stale'
    assert [x[0] for x in runner.calls]==['RUN_AGENT2.ps1']


def test_agent3_only_renders_current_timeline(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Render Only',str(script),str(audio),'existing',str(lib),'')
    Path(p.agent2_workspace,'timeline.json').write_text('{"audio":{"source":"fake","duration":3},"segments":[]}')
    p.stage_states.update({'match':'complete','render':'stale','done':'stale'}); ps.save(p)
    job=prepared_job(runner,reg,p,'agent3'); runner._run(p,job,'agent3')
    saved=ps.get(p.id)
    assert saved.status=='completed' and saved.stage_states['render']=='complete' and saved.stage_states['done']=='complete'
    assert [x[0] for x in runner.calls]==['RUN_AGENT3.ps1']


def test_agent1_only_invalidates_downstream(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); footage=tmp_path/'footage'; footage.mkdir()
    p=ps.create('Index Only',str(script),str(audio),'new','',str(footage))
    p.stage_states.update({'match':'complete','render':'complete','done':'complete'}); ps.save(p)
    job=prepared_job(runner,reg,p,'agent1'); runner._run(p,job,'agent1')
    saved=ps.get(p.id)
    assert saved.status=='stale' and saved.stage_states['index']=='complete'
    assert saved.stage_states['match']=='stale' and saved.stage_states['render']=='stale'
    assert [x[0] for x in runner.calls]==['RUN_AGENT1.ps1']


def test_pipeline_snapshot_survives_registry_restart(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Persistent Pipeline',str(script),str(audio),'existing',str(lib),'')
    job=prepared_job(runner,reg,p,'agent2'); runner._log(job,'persist me'); runner._run(p,job,'agent2')
    # New registry simulates restarting GhostCut.
    runner2=PipelineRunner(s,ps,JobRegistry()); saved=ps.get(p.id); snap=runner2.project_snapshot(saved)
    assert snap['stage_states']['match']=='complete'
    assert any('persist me' in x for x in snap['logs'])


def test_orphaned_persisted_running_job_is_recovered_and_unlocks_reruns(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Orphaned',str(script),str(audio),'existing',str(lib),'')
    # Reproduce a browser/server restart after project.json was left active.
    p.last_job_id='job_old'; p.status='running'; p.last_job_status='running'; p.pipeline_stage='preflight'; p.pipeline_stage_label='Preparing project'; p.pipeline_progress=3
    p.pipeline_message='Checking project inputs and agent workspaces'
    p.stage_states.update({'preflight':'complete','index':'complete','face':'complete','match':'complete','render':'complete','done':'complete'})
    ps.save(p)
    saved=ps.get(p.id); snap=runner.project_snapshot(saved)
    assert snap['status']=='interrupted'
    assert snap['stage_label']=='Interrupted'
    assert ps.get(p.id).status=='interrupted'
    # Most importantly, the synthetic snapshot is no longer treated as active by the UI.
    assert snap['status'] not in {'queued','running'}


def test_registered_live_worker_is_not_recovered_as_orphan(tmp_path):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Live',str(script),str(audio),'existing',str(lib),'')
    states=runner._prepare_run(p,'agent2'); job=reg.create(p.id,'agent2',states,str(Path(p.root)/'logs'/'ghostcut_pipeline.log'))
    thread=__import__('threading').Thread(target=lambda: None)
    reg.register_thread(job.id,thread)  # registered-but-not-started is a reserved live worker
    p.last_job_id=job.id; p.status='queued'; p.last_job_status='queued'; ps.save(p)
    out=runner.reconcile_project(ps.get(p.id))
    assert out.status=='queued'
    assert reg.active_job_id==job.id
    reg.finish_slot(job.id)


def test_start_save_failure_does_not_leave_phantom_active_slot(tmp_path, monkeypatch):
    s=settings(tmp_path); ps=ProjectStore(s); reg=JobRegistry(); runner=FakeRunner(s,ps,reg)
    script=tmp_path/'s.txt'; script.write_text('x'); audio=tmp_path/'a.mp3'; audio.write_bytes(b'x'); lib=tmp_path/'indexes'/'lib'; face_db(lib)
    p=ps.create('Save Failure',str(script),str(audio),'existing',str(lib),'')
    monkeypatch.setattr(ps,'save',lambda project: (_ for _ in ()).throw(PermissionError(5,'Access denied')))
    import pytest
    with pytest.raises(PermissionError): runner.start(p,'agent2')
    assert reg.active_job_id==''
    assert reg.threads=={}
