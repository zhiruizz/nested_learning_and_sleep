"""Build an isolated CLINC150 train/development pack from pinned local HF shards."""
from __future__ import annotations
import argparse, hashlib, json, random, unicodedata
from pathlib import Path
from typing import Iterable, Mapping

DATASET="clinc/clinc_oos"; REVISION="155b9c710419136e17307b80d0a13e68cd46b4ec"; CONFIG="plus"
SPECS={"train":(312096,"30188119cf9f86fc9db27e1c22442d091cb5cb0913c9496f945fe11e7a02a28f"),"validation":(77789,"fbd545b46c611c4a7ba4b48cae6c7f09bb5b59f33ff56206ad1cd366c85cdfaa")}
EXPECTED={"train":(15000,250,100),"validation":(3000,100,20)}
def canon(s): return "_".join(unicodedata.normalize("NFKC",s).casefold().strip().split())
def sha(b): return hashlib.sha256(b).hexdigest()
def class_orders(classes):
    out={}
    for seed in (17,29,43):
        ids=list(range(len(classes))); random.Random(seed).shuffle(ids); out[str(seed)]=ids
    return out
def read_label_names(meta):
    obj=json.loads(Path(meta).read_text(encoding="utf-8-sig"))
    if obj.get("id") != DATASET or obj.get("sha") != REVISION:
        raise ValueError("metadata dataset/revision mismatch")
    infos=obj["cardData"]["dataset_info"]
    info=next(x for x in infos if x["config_name"] == CONFIG)
    feature=next(x for x in info["features"] if x["name"] == "intent")
    names=feature["dtype"]["class_label"]["names"]
    if isinstance(names,dict):
        if set(names) != {str(i) for i in range(151)}:
            raise ValueError("expected contiguous 151 label IDs")
        names=[names[str(i)] for i in range(151)]
    if len(names)!=151 or len(set(names))!=151 or names[42]!="oos":
        raise ValueError("unexpected fixed plus label mapping")
    return names

def read_classes(meta):
    labels=read_label_names(meta)
    classes=sorted(canon(x) for x in labels if canon(x)!="oos")
    if len(classes)!=150 or len(set(classes))!=150: raise ValueError("expected 150 canonical in-scope labels")
    return classes
def iter_rows(parquet):
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(parquet)
    idx=0
    for batch in pf.iter_batches(batch_size=512,columns=["text","intent"]):
        for row in batch.to_pylist():
            yield idx,row; idx+=1
def source_id(split,index,text,label):
    material=f"{DATASET}\0{REVISION}\0{CONFIG}\0{split}\0{index}\0{text}\0{label}".encode("utf-8")
    return sha(material)

def normalized_text_hash(text):
    normalized=" ".join(unicodedata.normalize("NFKC",text).casefold().strip().split())
    return sha(normalized.encode("utf-8"))

def sha_file(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()
def pack(local_dir,metadata,out_dir):
    local_dir=Path(local_dir); out_dir=Path(out_dir)
    if out_dir.exists(): raise FileExistsError("output directory exists; refusing overwrite")
    out_dir.mkdir(parents=True)
    try:
        labels=read_label_names(metadata)
        classes=read_classes(metadata); class_id={n:i for i,n in enumerate(classes)}
        files={}; overlap=set(); train_hashes=set(); raw_overlap=set(); raw_train_hashes=set(); oos={}
        overlap_rows=0; raw_overlap_rows=0
        for split in ("train","validation"):
            p=local_dir/f"{split}-00000-of-00001.parquet"
            size,digest=SPECS[split]
            if p.stat().st_size!=size or sha_file(p)!=digest: raise ValueError(f"{split} shard size/hash mismatch")
            outname="train.jsonl" if split=="train" else "development.jsonl"
            h=hashlib.sha256(); counts={}; rows=0; oosn=0
            with (out_dir/outname).open("w",encoding="utf-8",newline="\n") as f:
                for index,row in iter_rows(p):
                    intent=row["intent"]; text=row["text"]
                    if type(intent) is not int or not 0 <= intent < len(labels):
                        raise ValueError("source intent must be fixed-mapping integer ID")
                    if not isinstance(text,str) or not text.strip():
                        raise ValueError("source text must be a nonempty string")
                    label=labels[intent]
                    if canon(label)=="oos": oosn+=1; continue
                    label=canon(label)
                    if label not in class_id: raise ValueError("row label absent from metadata classes")
                    tid=sha(text.encode("utf-8")); rid=source_id(split,index,text,label)
                    item={"id":rid,"text":text,"intent_id":class_id[label],"text_sha256":tid}
                    line=json.dumps(item,ensure_ascii=False,sort_keys=True)+"\n"; f.write(line); h.update(line.encode())
                    counts[label]=counts.get(label,0)+1; rows+=1
                    normalized_hash=normalized_text_hash(text)
                    if split=="train":
                        train_hashes.add(normalized_hash); raw_train_hashes.add(tid)
                    else:
                        if normalized_hash in train_hashes:
                            overlap.add(normalized_hash); overlap_rows+=1
                        if tid in raw_train_hashes:
                            raw_overlap.add(tid); raw_overlap_rows+=1
            exrows,exoos,perclass=EXPECTED[split]
            if rows!=exrows or oosn!=exoos or set(counts)!=set(classes) or set(counts.values())!={perclass}:
                raise ValueError(f"{split} distribution mismatch rows={rows},oos={oosn}")
            files["train" if split=="train" else "development"]={"path":outname,"sha256":h.hexdigest(),"rows":rows}
            oos[split]=oosn
        source={"provider":"huggingface","repo":DATASET,"config":CONFIG,"revision":REVISION,"license":"cc-by-3.0","url":"https://huggingface.co/datasets/clinc/clinc_oos"}
        source["metadata_sha256"]=sha_file(Path(metadata))
        source["raw_shards"]={split:{"filename":f"{split}-00000-of-00001.parquet","bytes":v[0],"sha256":v[1]} for split,v in SPECS.items()}
        manifest={"schema":"clinc-cil-training-manifest-v8","source":source,"classes":classes,"class_orders":class_orders(classes),"files":files,"normalization":{"unicode":"NFKC","case":"casefold","strip":True,"underscore_space_equivalent":True},"test_data_included":False,"oos_excluded":oos,"train_development_text_overlap":{"normalization":"NFKC,casefold,strip,collapse whitespace","count":len(overlap),"development_rows":overlap_rows,"sha256_sorted_ids":sha("\n".join(sorted(overlap)).encode()),"raw_count":len(raw_overlap),"raw_development_rows":raw_overlap_rows}}
        (out_dir/"pack-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
        return manifest
    except Exception as e:
        (out_dir/"FAILURE.json").write_text(json.dumps({"complete":False,"error_type":type(e).__name__,"error":str(e)},indent=2)+"\n",encoding="utf-8")
        raise
def main():
    p=argparse.ArgumentParser(); p.add_argument("--local-parquet-dir",required=True,type=Path); p.add_argument("--source-metadata",required=True,type=Path); p.add_argument("--out-dir",required=True,type=Path); a=p.parse_args(); pack(a.local_parquet_dir,a.source_metadata,a.out_dir)
if __name__=="__main__": main()
