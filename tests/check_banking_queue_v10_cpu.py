"""Finite launcher tests use a dummy process and never import a model framework."""
import hashlib,json,subprocess,sys
from pathlib import Path

RUNNER=Path(__file__).resolve().parents[1]/'tools'/'run_banking_queue_v10.py'

def fixture(tmp_path,status='qualified'):
    driver=tmp_path/'dummy_driver.py'
    driver.write_text("import json,sys\nfrom pathlib import Path\na=sys.argv; out=Path(a[a.index('--output-dir')+1]); out.mkdir()\n"+
      f"(out/'foundation-receipt.json').write_text(json.dumps({{'status':{status!r}}}))\n",encoding='utf-8')
    cfg=tmp_path/'config.json';cfg.write_text('{}')
    root=tmp_path/'queue'
    q={'schema':'banking-v10-immediate-queue-v1','queue_root':str(root),'driver':str(driver),
       'driver_sha256':hashlib.sha256(driver.read_bytes()).hexdigest(),'config':str(cfg),
       'config_file_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),
       'project_root':str(tmp_path),'model_path':str(tmp_path/'model'),
       'training_pack':str(tmp_path/'pack.json'),'jobs':[
          {'phase':'foundation','seed':17,'output_dir':str(root/'foundations'/'seed17')},
          {'phase':'foundation','seed':29,'output_dir':str(root/'foundations'/'seed29')}]}
    return q,root

def invoke(tmp_path,q):
    path=tmp_path/'queue.json';path.write_text(json.dumps(q))
    return subprocess.run([sys.executable,str(RUNNER),'--queue',str(path)],capture_output=True,text=True)

def test_negative_foundation_gate_stops_next_process(tmp_path):
    q,root=fixture(tmp_path,'failed_gate');result=invoke(tmp_path,q)
    assert result.returncode==2
    status=json.loads((root/'queue-status.json').read_text())
    assert status['status']=='stopped'and len(status['jobs'])==1
    assert not(root/'foundations'/'seed29').exists()

def test_changed_frozen_input_stops_before_creating_queue(tmp_path):
    q,root=fixture(tmp_path);Path(q['config']).write_text('{"changed":true}')
    result=invoke(tmp_path,q)
    assert result.returncode!=0 and not root.exists()

def test_output_path_cannot_escape_fresh_namespace(tmp_path):
    q,root=fixture(tmp_path);q['jobs'][1]['output_dir']=str(tmp_path/'escape')
    result=invoke(tmp_path,q)
    assert result.returncode!=0 and not root.exists()and not(tmp_path/'escape').exists()

def test_storage_threshold_fails_before_process_or_directory(tmp_path):
    q,root=fixture(tmp_path);q['minimum_free_before_start_bytes']=10**30
    result=invoke(tmp_path,q)
    assert result.returncode!=0 and not root.exists()
