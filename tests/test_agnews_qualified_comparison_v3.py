from pathlib import Path
from copy import deepcopy
import sys,pytest
pytest.importorskip('transformers')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_agnews_qualified_comparison_v3 as run
import agnews_v3_common as c

def scores(phase):
 ids=(0,1)if phase=='old'else(2,3)
 return {phase:{c.EM:.90},**{f'class{i}':{c.EM:.90}for i in ids}}
def fixture():
 old=[{'seed':s,'train':scores('old'),'validation':scores('old')}for s in c.SEEDS]
 new={str(s):{'train':scores('new'),'validation':scores('new')}for s in c.SEEDS}
 return {'all_qualified':True,'records':old},{'old_all_qualified':True,'new_sft_all_qualified':True,'new_sft_records':new,'new_steps':512},{'new_steps':512}
def test_all_seeds_and_perclass_gate_before_comparison():
 cert,q,cfg=fixture();run.validate_qualification(cert,q,cfg)
 q['new_sft_records']['29']['validation']['class2'][c.EM]=.69
 with pytest.raises(AssertionError):run.validate_qualification(cert,q,cfg)
 cert,q,cfg=fixture();cert['records'].pop()
 with pytest.raises(AssertionError):run.validate_qualification(cert,q,cfg)
def test_budgets_have_enough_slots_and_terminal_clocks():
 for n in (512,1024,2048):
  assert n//64<=c.CAPACITY and(n+12*(n//64))%16==0
  assert[n//4,n//2,3*n//4,n]==sorted({n//4,n//2,3*n//4,n})
