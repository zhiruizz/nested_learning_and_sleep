import hashlib
from tools.prepare_banking_train_v10 import split_records,normalize_text

def row(i,text,label):return {'id':hashlib.sha256(str(i).encode()).hexdigest(),'text':text,'source_label':label,'source_row':i}
def fixture():return [row(i+1000*c,f'class {c} request {i}',f'label_{c}') for c in range(3) for i in range(60)]
def test_normalized_groups_never_cross_split_and_duplicates_not_exposure():
    data=fixture();data.append(row(9000,' CLASS 0   REQUEST 0 ','label_0'))
    classes,train,dev,audit=split_records(data,dev_per_class=10)
    assert len(train)==150 and len(dev)==30 and audit['same_label_duplicate_rows_excluded']==1
    assert {normalize_text(x['text']) for x in train}.isdisjoint({normalize_text(x['text']) for x in dev})
def test_conflicting_groups_fully_quarantined_before_sampling():
    data=fixture()+[row(9900,'conflicting phrase','label_0'),row(9901,'CONFLICTING phrase','label_1')]
    _,train,dev,audit=split_records(data,dev_per_class=10)
    assert audit['conflicting_text_groups_quarantined']==1 and audit['conflicting_source_rows_quarantined']==2
    assert all(normalize_text(r['text'])!='conflicting phrase' for r in train+dev)
def test_reordering_source_does_not_change_frozen_sample_membership():
    data=fixture()
    a=split_records(data,dev_per_class=10)
    b=split_records(list(reversed(data)),dev_per_class=10)
    assert [r['id'] for r in a[1]]==[r['id'] for r in b[1]]
    assert [r['id'] for r in a[2]]==[r['id'] for r in b[2]]

def test_imbalanced_classes_keep_all_remaining_rows():
    data=fixture()+[row(20000+i,f'extra {i}','label_0') for i in range(17)]
    _,train,dev,audit=split_records(data)
    assert len(train)==167 and len(dev)==30
    assert audit['train_counts_per_class']=={'label_0':67,'label_1':50,'label_2':50}
    assert audit['unused_unique_training_source_rows']==0

def test_smallest_original_class_can_reserve_ten_dev_without_discarding_train():
    data=[row(i,f'phrase {i}','single_class') for i in range(35)]
    _,train,dev,_=split_records(data)
    assert len(train)==25 and len(dev)==10
