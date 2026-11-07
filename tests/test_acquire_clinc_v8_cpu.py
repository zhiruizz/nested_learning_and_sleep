import importlib.util, json, hashlib
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

P=Path(__file__).parents[1]/"tools"/"acquire_clinc_v8.py"
spec=importlib.util.spec_from_file_location("acq_clinc_v8",P); acq=importlib.util.module_from_spec(spec); spec.loader.exec_module(acq)

def fixture(tmp_path, monkeypatch):
    labels=[f"intent_{i:03}" for i in range(151)]; labels[42]="oos"
    metadata={"id":acq.DATASET,"sha":acq.REVISION,"cardData":{"dataset_info":[{"config_name":"plus","features":[{"name":"text","dtype":"string"},{"name":"intent","dtype":{"class_label":{"names":{str(i):v for i,v in enumerate(labels)}}}}]}]}}
    mp=tmp_path/"metadata.json"; mp.write_text(json.dumps(metadata),encoding="utf-8")
    d=tmp_path/"shards"; d.mkdir()
    for split, nper, oosn in [("train",2,2),("validation",1,1)]:
        texts=[]; ids=[]
        for i in [x for x in range(151) if x != 42]:
            for j in range(nper):
                texts.append(" Café  Hello " if split=="train" and i==0 else ("cafe\u0301 hello" if split=="validation" and i==0 else f"{split}-{i}-{j}")); ids.append(i)
        texts += [f"OOS example {i}" for i in range(oosn)]; ids += [42]*oosn
        path=d/f"{split}-00000-of-00001.parquet"; pq.write_table(pa.table({"text":texts,"intent":ids}),path)
        data=path.read_bytes(); monkeypatch.setitem(acq.SPECS,split,(len(data),hashlib.sha256(data).hexdigest()))
        monkeypatch.setitem(acq.EXPECTED,split,(150*nper,oosn,nper))
    # Test shard exists to prove pack never opens it.
    (d/"test-00000-of-00001.parquet").write_text("DO NOT READ")
    return d,mp

def test_end_to_end_real_metadata_and_integer_labels(tmp_path,monkeypatch):
    d,m=fixture(tmp_path,monkeypatch); out=tmp_path/"pack"
    result=acq.pack(d,m,out)
    assert result["schema"]=="clinc-cil-training-manifest-v8"
    assert result["files"]["train"]["rows"]==300 and result["files"]["development"]["rows"]==150
    assert result["oos_excluded"]=={"train":2,"validation":1}
    assert result["train_development_text_overlap"]["count"]==1
    assert result["train_development_text_overlap"]["raw_count"]==0
    rows=[json.loads(x) for x in (out/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"]==" Café  Hello "
    assert len({r["id"] for r in rows})==len(rows)
    assert "test" not in result["files"] and result["test_data_included"] is False
    assert len(result["class_orders"]["17"])==150
    from tools.clinc_cil_common_v8 import validate_manifest
    validate_manifest(result)

def test_bad_hash_failure_and_no_complete_manifest(tmp_path,monkeypatch):
    d,m=fixture(tmp_path,monkeypatch); monkeypatch.setitem(acq.SPECS,"train",(0,"bad")); out=tmp_path/"badpack"
    with pytest.raises(ValueError): acq.pack(d,m,out)
    assert (out/"FAILURE.json").exists() and not (out/"pack-manifest.json").exists()

def test_refuses_overwrite(tmp_path,monkeypatch):
    d,m=fixture(tmp_path,monkeypatch); out=tmp_path/"existing"; out.mkdir(); (out/"sentinel").write_text("x")
    with pytest.raises(FileExistsError): acq.pack(d,m,out)
    assert (out/"sentinel").read_text()=="x"
