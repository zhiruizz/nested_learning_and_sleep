from pathlib import Path
import sys,pytest
pytest.importorskip('transformers')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_agnews_preupdate_comparison_v4 as driver

def test_event_pools_are_balanced_train_only_and_reproducible():
 pack={'records':[{'class_id':k,'role':role}for k in range(4)for role in ['train']*8+['validation','holdout']]}
 first=driver.event_pools(pack,17);again=driver.event_pools(pack,17);other=driver.event_pools(pack,29)
 assert first==again and first!=other and len(first)==28
 for pool in first:
  assert len(pool)==24 and all(pack['records'][i]['role']=='train'for i in pool)
  assert [sum(pack['records'][i]['class_id']==k for i in pool)for k in range(4)]==[6,6,6,6]
  assert [sum(pack['records'][i]['class_id']==k for i in pool[:16])for k in range(4)]==[4,4,4,4]
  assert [sum(pack['records'][i]['class_id']==k for i in pool[16:])for k in range(4)]==[2,2,2,2]
