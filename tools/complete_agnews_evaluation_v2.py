"""Repeat frozen read-only scoring, verifying and preserving interrupted outputs.

No training, new test selection, new metric, overwritten score or checkpoint edit.
"""
from datetime import datetime,timezone,timedelta
import hashlib,importlib.util,json
from pathlib import Path
import sys,time
from types import SimpleNamespace

ROOT=Path('D:/my_paper/NL_Sleep_Public_Candidate')
RUN=ROOT/'outputs/agnews-retention-v2/main-v2-seeds17-29-43'
OUT=RUN/'evaluation-v2'
SOURCE=ROOT/'tools/evaluate_agnews_retention_v2.py'
EXPECTED='b09cc5de75b60cf664472537b6eeee28cb391d68d305218f99fbf91ed51c234d'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
assert not (OUT/'evaluation-summary.json').exists()
spec=importlib.util.spec_from_file_location('frozen_eval_v2',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
seal,seal_hash=module.require_seal(RUN)
deadline=datetime.fromisoformat(json.loads((RUN/'training-protocol.json').read_text(encoding='utf-8'))['sealed_utc'])+timedelta(hours=1)
existing={p.name:{'sha256':module.sha(p),'value':json.loads(p.read_text(encoding='utf-8'))}
          for p in OUT.glob('score-*.json')}
verified=[];original=module.write_new
receipt={'schema':'agnews-readonly-evaluation-completion-v2','status':'running',
 'reason':'initial worker/session ended unexpectedly; no Python process remained, no accessible exit traceback',
 'original_completed_checkpoint_scores':len(existing),'original_exit_cause':'unknown',
 'original_scores_sha256':{n:r['sha256']for n,r in existing.items()},
 'frozen_evaluator_sha256':EXPECTED,'completion_wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'same_test_pack_sha256':module.sha(OUT/'test-pack-private.json'),'seal_sha256':seal_hash,
 'new_test_selection':False,'training_resumed':False,'overwrite_existing_scores':False,
 'started_utc':datetime.now(timezone.utc).isoformat(),
 'first_pass_total_work':'not fully persisted; 32 complete checkpoint scores and two foundation qualifications give a lower bound; one in-progress checkpoint may add up to 64 batches',
 'same_absolute_evaluation_deadline_utc':deadline.isoformat()}
original(OUT/'completion-receipt-start.json',receipt)


def preserve_or_write(path,value):
    if datetime.now(timezone.utc)>=deadline:
        raise TimeoutError('combined post-seal evaluation one-hour cap reached')
    if path.exists():
        if path.name not in existing or json.loads(path.read_text(encoding='utf-8'))!=value:
            raise RuntimeError('existing score differs from repeated frozen computation; refusing overwrite: '+path.name)
        if module.sha(path)!=existing[path.name]['sha256']:
            raise RuntimeError('existing score file changed during repeat')
        verified.append(path.name)
        return
    original(path,value)


module.write_new=preserve_or_write
started=time.monotonic()
try:
    module.score_test(SimpleNamespace(run_dir=RUN,out_dir=OUT,
        records=ROOT/'outputs/agnews-retention-v1/data/agnews-train-roles-v1.json',
        model_dir=Path('D:/my_paper/Nested_Learning_and_Sleep/.cache/models/llama32-3b')),seal,seal_hash)
    assert set(verified)==set(existing)
    receipt.update(status='complete',repeated_scores_identical=len(verified),seconds=time.monotonic()-started,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    original(OUT/'completion-receipt-final.json',receipt)
    print(json.dumps({'status':'completion_success','existing_scores_verified_identical':len(verified)}),flush=True)
except BaseException as exc:
    receipt.update(status='failed',error_type=type(exc).__name__,error=str(exc),
                   repeated_scores_identical=len(verified),seconds=time.monotonic()-started)
    original(OUT/'completion-receipt-failure.json',receipt)
    raise
