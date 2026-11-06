from pathlib import Path
import json,sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import summarize_agnews_sleep_forensics_v3 as summary
ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/agnews-sleep-forensics-v3'

def test_bundled_diagnostic_reanalysis_is_identical():
 result=summary.summarize(RESULTS/'diagnostic-result.json',RESULTS/'diagnostic-plan.json')
 assert result==json.loads((RESULTS/'analysis.json').read_text(encoding='utf-8'))
 assert result['independent_greedy_checks']['comparisons']==864

def test_corrupt_provenance_or_validation_denominator_is_rejected(tmp_path):
 data=json.loads((RESULTS/'diagnostic-result.json').read_text(encoding='utf-8'));dest=tmp_path/'changed.json'
 data['heldout_read']=True;dest.write_text(json.dumps(data),encoding='utf-8')
 with pytest.raises(AssertionError):summary.summarize(dest,RESULTS/'diagnostic-plan.json')
 data['heldout_read']=False;data['source_sha256']='wrong';dest.write_text(json.dumps(data),encoding='utf-8')
 with pytest.raises(AssertionError):summary.summarize(dest,RESULTS/'diagnostic-plan.json')
 data['source_sha256']=json.loads((RESULTS/'diagnostic-plan.json').read_text())['source_sha256']
 data['results'][0]['validation_trajectory_and_first_boundary'][0]['metrics']['old']['examples']=512
 dest.write_text(json.dumps(data),encoding='utf-8')
 with pytest.raises(AssertionError):summary.summarize(dest,RESULTS/'diagnostic-plan.json')
