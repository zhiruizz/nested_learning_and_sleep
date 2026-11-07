"""Run an immediate finite serial queue; no schedules, retries, or GPU fallback."""
from __future__ import annotations
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n',encoding='utf-8',newline='\n')
    os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue',type=Path,required=True)
    a=p.parse_args();q=json.loads(a.queue.read_text(encoding='utf-8'))
    if q['schema']!='banking-v10-immediate-queue-v1':raise ValueError('Unknown queue schema')
    root=Path(q['queue_root']);status=root/'queue-status.json'
    if root.exists():raise FileExistsError('Queue directory exists: inspect current state; never auto-retry')
    jobs=q['jobs']
    if not jobs or len(jobs)>9:raise ValueError('Queue must be finite, one to nine jobs')
    planned=[Path(job['output_dir']).resolve() for job in jobs]
    if len(set(planned))!=len(planned) or any(root.resolve() not in out.parents for out in planned):
        raise ValueError('Every output must be a unique child of this fresh queue directory')
    driver=Path(q['driver']);config=Path(q['config'])
    expected_driver=q['driver_sha256'];expected_config=q['config_file_sha256']
    if file_sha(driver)!=expected_driver or file_sha(config)!=expected_config:raise ValueError('Frozen launch inputs changed')
    minimum_free=q.get('minimum_free_before_start_bytes',0)
    if type(minimum_free)is not int or minimum_free<0:raise ValueError('Invalid storage threshold')
    free_before=shutil.disk_usage(root.parent).free
    if free_before<minimum_free:raise RuntimeError('Insufficient free space for this frozen queue')
    root.mkdir(parents=False)
    state={'schema':'banking-v10-immediate-queue-status-v1','status':'running','started_utc':now(),
           'queue_sha256':file_sha(a.queue),'pid':os.getpid(),'jobs':[],'test_payload_accessed':False,'automatic_retries':0,
           'launcher_sha256':file_sha(Path(__file__)),'free_bytes_before_start':free_before,
           'minimum_free_before_start_bytes':minimum_free}
    save(status,state)
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env['TOKENIZERS_PARALLELISM']='false'
    env['PYTHONPATH']=str(Path(q['project_root'])/'src')+os.pathsep+str(driver.parent.parent)
    for job in jobs:
        if file_sha(driver)!=expected_driver or file_sha(config)!=expected_config:raise ValueError('Source/config changed between jobs')
        phase=job['phase']
        if phase not in ('foundation','main'):raise ValueError('Unsupported queue phase')
        out=Path(job['output_dir'])
        if out.exists():raise FileExistsError(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(driver),'--phase',phase,'--project-root',q['project_root'],
             '--model-path',q['model_path'],'--training-pack',q['training_pack'],
             '--config',str(config),'--output-dir',str(out),'--seed',str(job['seed'])]
        if phase=='main':
            cmd+=['--arm',job['arm'],'--foundation-dir',job['foundation_dir'],'--release-artifact',q['release_artifact']]
        label=f"{phase}-seed{job['seed']}"+(f"-{job['arm']}" if 'arm'in job else '')
        record={'job':label,'output_dir':str(out),'started_utc':now(),'status':'running'}
        state['jobs'].append(record);save(status,state)
        print(json.dumps({'status':'starting','job':label,'utc':now()}),flush=True)
        started=time.monotonic()
        with (root/f'{label}.stdout.log').open('xb') as stdout,(root/f'{label}.stderr.log').open('xb') as stderr:
            result=subprocess.run(cmd,stdout=stdout,stderr=stderr,env=env,cwd=driver.parent.parent,check=False)
        record.update(returncode=result.returncode,elapsed_seconds=time.monotonic()-started,ended_utc=now())
        receipt=out/('foundation-receipt.json' if phase=='foundation' else 'run-receipt.json')
        valid=False
        if result.returncode==0 and receipt.is_file():
            received=json.loads(receipt.read_text(encoding='utf-8'));valid=received.get('status')==('qualified'if phase=='foundation'else'complete')
            record.update(receipt_sha256=file_sha(receipt),receipt_status=received.get('status'))
        record['status']='complete'if valid else'stopped'
        print(json.dumps(record),flush=True)
        if not valid:
            state.update(status='stopped',ended_utc=now());save(status,state)
            raise SystemExit(result.returncode or 2)
        save(status,state)
    state.update(status='complete',ended_utc=now());save(status,state)
    print(json.dumps({'status':'complete','jobs':len(jobs),'utc':now()}),flush=True)
if __name__=='__main__':main()
