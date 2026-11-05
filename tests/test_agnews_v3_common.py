from pathlib import Path
import sys,pytest
pytest.importorskip('transformers')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
import agnews_v3_common as c

def metric(phase,overall,a,b):
 classes=[0,1]if phase=='old'else[2,3]
 return {phase:{c.EM:overall},f'class{classes[0]}':{c.EM:a},f'class{classes[1]}':{c.EM:b}}
def test_global_qualification_requires_both_validation_classes():
 assert c.qualifies(metric('old',.8,.8,.8),metric('old',.75,.7,.8))
 assert not c.qualifies(metric('old',.99,.99,.99),metric('old',.8,.69,.91))
 assert not c.qualifies(metric('new',.79,.79,.79),metric('new',.99,.99,.99),'new')
def test_batch_only_exposes_prompt_and_shared_marker_not_class_target():
 pack={'records':[{'tokens':[1,2,3],'class_id':x}for x in range(4)],
  'code_maps':{'17':{'marker_token_id':11,'class_to_token_ids':[24,23,22,21]}},'tokenizer':{'pad_token_id':0}}
 ids,mask,lengths,markers,labels=c.batch(pack,17,[0,1,2,3],device='cpu')
 assert ids.tolist()==[[1,2,3,11]]*4
 assert labels.tolist()==[24,23,22,21] and markers.tolist()==[11]*4
 assert lengths.tolist()==[3]*4 and mask.sum().item()==16
