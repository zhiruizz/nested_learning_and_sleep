"""Pure-Python data and metric helpers for 10x15 CLINC intent CIL.

This module never loads a model or tokenizer implementation. Tokenization is
provided by the caller. Generation is unconstrained; the intent list is used
only to normalize/score decoded predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import unicodedata
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

SCHEMA="clinc-cil-training-manifest-v8"
SEEDS=(17,29,43)
TASKS=10
CLASSES_PER_TASK=15
NUM_CLASSES=150
PROMPT_TEMPLATE="Classify the intent of this user request.\nRequest: {text}\nIntent:"
PROMPT_HEAD="Classify the intent of this user request.\nRequest: "
PROMPT_TAIL="\nIntent:"
IGNORE_INDEX=-100

@dataclass(frozen=True)
class TrainingView:
    """Manifest subset usable during training; deliberately has no test field."""
    schema: str
    classes: tuple[str,...]
    class_orders: Mapping[str,tuple[int,...]]
    train: Mapping[str,object]
    development: Mapping[str,object]
    source_sha256: str

def sha256_file(path: str|Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def canonical_json_sha256(value) -> str:
    b=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def _validate_classes(classes: Sequence[str]) -> tuple[str,...]:
    xs=tuple(classes)
    if len(xs)!=NUM_CLASSES or len(set(xs))!=NUM_CLASSES: raise ValueError("expected 150 unique canonical intent names")
    if any(not isinstance(x,str) or not x or x!=x.strip() or x.lower()!=x for x in xs): raise ValueError("class names must be lowercase canonical strings")
    if xs!=tuple(sorted(xs)): raise ValueError("canonical classes must be lexicographically sorted")
    return xs

def validate_manifest(manifest: Mapping) -> None:
    if manifest.get("schema")!=SCHEMA: raise ValueError("manifest schema mismatch")
    classes=_validate_classes(manifest.get("classes",()))
    orders=manifest.get("class_orders")
    if not isinstance(orders,dict) or set(orders)!={str(s) for s in SEEDS}: raise ValueError("class_orders must define seeds 17, 29, 43")
    expected=set(range(NUM_CLASSES))
    for seed in SEEDS:
        order=orders.get(str(seed),orders.get(seed))
        if len(order)!=NUM_CLASSES: raise ValueError(f"seed {seed}: class order must contain 150 entries")
        if any(not isinstance(x,int) for x in order) or set(order)!=expected: raise ValueError(f"seed {seed}: class order must be a permutation of 0..149")
    files=manifest.get("files")
    if not isinstance(files,dict) or set(files)!={"train","development"}: raise ValueError("training manifest must contain only train and development files")
    if manifest.get("test_data_included") is not False: raise ValueError("training manifest must explicitly exclude test data")
    source=manifest.get("source")
    if not isinstance(source,dict) or not isinstance(source.get("metadata_sha256"),str) or len(source["metadata_sha256"])!=64:
        raise ValueError("manifest source metadata hash is required")
    for role,meta in files.items():
        if not isinstance(meta,dict) or not isinstance(meta.get("path"),str) or not isinstance(meta.get("sha256"),str) or len(meta["sha256"])!=64 or not isinstance(meta.get("rows"),int):
            raise ValueError(f"malformed {role} file metadata")
        if meta["rows"]<0: raise ValueError("row count cannot be negative")
    if manifest.get("normalization")!={"unicode":"NFKC","case":"casefold","strip":True,"underscore_space_equivalent":True}:
        raise ValueError("normalization must be explicitly locked")
    # Ensure canonical names are parseable and unambiguous under the locked map.
    normalized=[normalize_prediction(x) for x in classes]
    if len(set(normalized))!=NUM_CLASSES: raise ValueError("canonical labels collide after normalization")

def make_training_view(manifest: Mapping) -> TrainingView:
    """Return only train/dev paths; test metadata is intentionally stripped."""
    validate_manifest(manifest)
    classes=tuple(manifest["classes"]); names_to_id={name:i for i,name in enumerate(classes)}
    raw_orders=manifest["class_orders"]
    orders={str(k):tuple(int(x) for x in v) for k,v in raw_orders.items()}
    return TrainingView(SCHEMA,classes,
        orders,
        dict(manifest["files"]["train"]),dict(manifest["files"]["development"]),
        canonical_json_sha256(manifest["source"]))

def resolve_role_path(view: TrainingView, role: str, root: str|Path) -> Path:
    if role not in ("train","development"): raise ValueError("training view only exposes train and development")
    meta=getattr(view,role)
    return Path(root)/str(meta["path"])

def _validate_row(row: Mapping, classes: Sequence[str], expected_role: str|None=None) -> None:
    row_id=row.get("id")
    if not isinstance(row_id,str) or not row_id: raise ValueError("row requires id or record_id_sha256")
    if len(row_id)!=64 or any(ch not in "0123456789abcdef" for ch in row_id.lower()): raise ValueError("id must be source-bound hex64")
    if not isinstance(row.get("text"),str): raise ValueError("row text must be a string")
    iid=row.get("intent_id")
    if not isinstance(iid,int) or not 0<=iid<len(classes): raise ValueError("intent_id outside canonical class range")
    text_sha=row.get("text_sha256")
    if not isinstance(text_sha,str) or len(text_sha)!=64 or any(ch not in "0123456789abcdef" for ch in text_sha.lower()): raise ValueError("text_sha256 must be hex64")
    actual_text_sha=hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    if text_sha.lower()!=actual_text_sha: raise ValueError("text_sha256 does not match original UTF-8 text")
    if expected_role is not None and row.get("role",expected_role)!=expected_role: raise ValueError("row role mismatch")

def row_intent_id(row: Mapping, classes: Sequence[str]) -> int:
    value=row.get("intent_id")
    if not isinstance(value,int): raise ValueError("row lacks integer intent_id")
    return value

def iter_jsonl(path: str|Path, metadata: Mapping, classes: Sequence[str], *, role: str) -> Iterator[dict]:
    """Two-pass streaming reader: verify SHA first, then yield validated JSONL."""
    path=Path(path)
    actual=sha256_file(path)
    if actual!=metadata["sha256"]: raise ValueError(f"{role} file SHA256 mismatch")
    count=0; ids=set()
    with path.open("r",encoding="utf-8",newline="") as f:
        for line_no,line in enumerate(f,1):
            if not line.strip(): raise ValueError(f"blank JSONL line at {line_no}")
            row=json.loads(line);_validate_row(row,classes,role)
            row_id=row["id"]
            if row_id in ids: raise ValueError(f"duplicate {role} id: {row_id}")
            ids.add(row_id);count+=1;yield row
    if count!=metadata["rows"]: raise ValueError(f"{role} row count mismatch: {count} != {metadata['rows']}")

def iter_role(view: TrainingView, role: str, root: str|Path) -> Iterator[dict]:
    if role not in ("train","development"): raise ValueError("test is available only through the dedicated test reader")
    meta=getattr(view,role)
    return iter_jsonl(resolve_role_path(view,role,root),meta,view.classes,role=role)

def iter_test_after_seal(test_manifest: Mapping, root: str|Path, training_seal: Mapping, *,
                         expected_seed_arms: Iterable[tuple[int,str]],
                         expected_config_sha256: str,
                         expected_training_manifest_sha256: str) -> Iterator[dict]:
    """Dedicated test reader gated by the complete six-run seal and locked inputs."""
    if training_seal.get("complete") is not True or training_seal.get("test_data_included") is not False:
        raise ValueError("test is unavailable until the training seal is complete")
    if test_manifest.get("schema")!="clinc-cil-test-manifest-v8": raise ValueError("test manifest schema mismatch")
    expected={(int(s),str(a)) for s,a in expected_seed_arms}
    if len(expected)!=6: raise ValueError("expected six unique seed/arm training runs")
    runs=training_seal.get("runs")
    if not isinstance(runs,list): raise ValueError("training seal lacks run list")
    completed={(int(r.get("seed",-1)),str(r.get("arm",""))) for r in runs if r.get("status")=="complete"}
    if len(runs)!=6 or completed!=expected: raise ValueError("training seal is not the complete expected seed/arm grid")
    for run in runs:
        if run.get("tasks_completed")!=TASKS: raise ValueError("each training run must complete all ten tasks")
        for field in ("final_checkpoint_sha256","run_receipt_sha256"):
            value=run.get(field)
            if not isinstance(value,str) or len(value)!=64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError(f"each training run requires valid {field}")
        if run.get("config_sha256")!=expected_config_sha256: raise ValueError("run config hash mismatch")
        if run.get("training_manifest_sha256")!=expected_training_manifest_sha256: raise ValueError("run manifest hash mismatch")
    if training_seal.get("config_sha256")!=expected_config_sha256: raise ValueError("training config hash mismatch")
    if training_seal.get("training_manifest_sha256")!=expected_training_manifest_sha256: raise ValueError("training manifest hash mismatch")
    if test_manifest.get("training_manifest_sha256")!=expected_training_manifest_sha256: raise ValueError("test manifest is bound to a different training manifest")
    classes=_validate_classes(test_manifest.get("classes",()))
    meta=test_manifest.get("test")
    if not isinstance(meta,dict): raise ValueError("test metadata missing")
    return iter_jsonl(Path(root)/meta["path"],meta,classes,role="test")

def validate_role_partition(train: Iterable[Mapping], development: Iterable[Mapping], test: Iterable[Mapping]|None=None,
                            *, train_per_class: int=100, development_per_class: int=20, test_per_class: int=30) -> dict:
    """Check complete split coverage and ID disjointness without retaining rows."""
    seen=set();counts={};totals={}
    for role,rows,expected in (("train",train,train_per_class),("development",development,development_per_class),("test",test,test_per_class)):
        if rows is None: continue
        cc=[0]*NUM_CLASSES;n=0
        for row in rows:
            rid=row.get("id",row.get("record_id_sha256"));iid=int(row["intent_id"])
            if rid in seen: raise ValueError("split row IDs overlap")
            seen.add(rid);cc[iid]+=1;n+=1
        if any(x!=expected for x in cc): raise ValueError(f"{role} must have {expected} rows per class")
        counts[role]=cc;totals[role]=n
    return {"rows_by_role":totals,"examples_per_class":counts}

def task_for_class(class_id: int, class_order: Sequence[int]) -> int:
    if len(class_order)!=NUM_CLASSES or class_id not in set(class_order): raise ValueError("class/order mismatch")
    return class_order.index(class_id)//CLASSES_PER_TASK

def task_class_ids(class_order: Sequence[int], task_id: int) -> tuple[int,...]:
    if len(class_order)!=NUM_CLASSES or not 0<=task_id<TASKS: raise ValueError("invalid class order or task id")
    start=task_id*CLASSES_PER_TASK
    return tuple(class_order[start:start+CLASSES_PER_TASK])

def iter_task_rows(rows: Iterable[Mapping], class_order: Sequence[int], task_id: int, classes: Sequence[str]|None=None) -> Iterator[Mapping]:
    allowed=set(task_class_ids(class_order,task_id))
    for row in rows:
        if row_intent_id(row,classes or ()) in allowed: yield row

def shuffled_ids(rows: Iterable[Mapping], seed: int, task_id: int) -> list[str]:
    """Stable deterministic schedule over row IDs, independent of process hash seed."""
    ids=[str(r.get("id",r.get("record_id_sha256"))) for r in rows]
    random.Random(f"clinc-cil-v8:{seed}:{task_id}").shuffle(ids)
    return ids

def build_prompt(text: str) -> str:
    if not isinstance(text,str): raise TypeError("text must be str")
    return PROMPT_TEMPLATE.format(text=text)

def _encode(tokenizer, text: str) -> list[int]:
    if hasattr(tokenizer,"encode"): out=tokenizer.encode(text,add_special_tokens=False)
    else: out=tokenizer(text,add_special_tokens=False)["input_ids"]
    if out and isinstance(out[0],list): out=out[0]
    return [int(x) for x in out]

def tokenize_prompt(text: str, tokenizer, classes: Sequence[str], eos_token_id: int,
                    max_length: int|None=None, *, prepend_bos: bool=False,
                    bos_token_id: int|None=None) -> dict:
    """Tokenize prompt alone, reserving capacity for the longest natural label+EOS.

    The returned prompt is gold-independent, so train/eval/generation prefixes
    for identical text are identical regardless of intent_id. Encoded segments
    are concatenated without automatic specials or inserted separators.
    """
    classes=_validate_classes(classes)
    if not isinstance(text,str): raise TypeError("text must be str")
    head_ids=_encode(tokenizer,PROMPT_HEAD);body_ids=_encode(tokenizer,text);tail_ids=_encode(tokenizer,PROMPT_TAIL)
    if prepend_bos:
        if bos_token_id is None: raise ValueError("explicit BOS requested without bos_token_id")
        head_ids=[int(bos_token_id)]+head_ids
    elif bos_token_id is not None: raise ValueError("pass bos_token_id only with prepend_bos=True")
    max_label_tokens=max(len(_encode(tokenizer,label)) for label in classes)
    if max_label_tokens<1: raise ValueError("canonical label tokenized to empty sequence")
    if max_length is not None:
        body_budget=int(max_length)-len(head_ids)-len(tail_ids)-max_label_tokens-1
        if body_budget<0: raise ValueError("max_length cannot fit prompt frame and longest label+EOS")
        if len(body_ids)>body_budget: body_ids=body_ids[:body_budget]
    prompt_ids=head_ids+body_ids+tail_ids
    return {"input_ids":prompt_ids,"attention_mask":[1]*len(prompt_ids),"prompt_length":len(prompt_ids)}

def tokenize_row(row: Mapping, tokenizer, classes: Sequence[str], eos_token_id: int, max_length: int|None=None,
                 *, prepend_bos: bool=False, bos_token_id: int|None=None) -> dict:
    """Tokenize an identical gold-independent prompt then append natural label+EOS."""
    _validate_row(row,classes)
    prefix=tokenize_prompt(row["text"],tokenizer,classes,eos_token_id,max_length,
        prepend_bos=prepend_bos,bos_token_id=bos_token_id)
    target_ids=_encode(tokenizer,classes[row_intent_id(row,classes)])
    if not target_ids: raise ValueError("canonical label tokenized to empty sequence")
    prompt_ids=prefix["input_ids"]
    input_ids=prompt_ids+target_ids+[int(eos_token_id)]
    labels=[IGNORE_INDEX]*len(prompt_ids)+target_ids+[int(eos_token_id)]
    return {"id":row["id"],"intent_id":row_intent_id(row,classes),"prompt_length":len(prompt_ids),"input_ids":input_ids,"labels":labels,"attention_mask":[1]*len(input_ids)}

def tokenize_batch(rows: Sequence[Mapping], tokenizer, classes: Sequence[str], eos_token_id: int, pad_token_id: int, max_length: int|None=None,
                   *, prepend_bos: bool=False, bos_token_id: int|None=None) -> dict:
    if not rows: raise ValueError("empty batch")
    encoded=[tokenize_row(r,tokenizer,classes,eos_token_id,max_length,prepend_bos=prepend_bos,bos_token_id=bos_token_id) for r in rows]
    width=max(len(x["input_ids"]) for x in encoded)
    return {"ids":[x["id"] for x in encoded],"intent_ids":[x["intent_id"] for x in encoded],
        "input_ids":[x["input_ids"]+[int(pad_token_id)]*(width-len(x["input_ids"])) for x in encoded],
        "attention_mask":[x["attention_mask"]+[0]*(width-len(x["attention_mask"])) for x in encoded],
        "labels":[x["labels"]+[IGNORE_INDEX]*(width-len(x["labels"])) for x in encoded],
        "prompt_lengths":[x["prompt_length"] for x in encoded]}

def normalize_prediction(text: str, *, underscore_space_equivalent: bool=True) -> str:
    value=unicodedata.normalize("NFKC",str(text)).casefold().strip()
    if underscore_space_equivalent: value=value.replace("_"," ")
    return " ".join(value.split())

def decode_label_prefix(token_ids: Sequence[int], tokenizer, eos_token_id: int) -> str:
    try: stop=list(token_ids).index(int(eos_token_id)); token_ids=token_ids[:stop]
    except ValueError: pass
    return tokenizer.decode(list(token_ids),skip_special_tokens=True)

def score_predictions(predictions: Sequence[str], gold_intent_ids: Sequence[int], classes: Sequence[str], class_order: Sequence[int]|None=None) -> dict:
    if len(predictions)!=len(gold_intent_ids): raise ValueError("prediction/gold length mismatch")
    classes=_validate_classes(classes); label_to_id={normalize_prediction(x):i for i,x in enumerate(classes)}
    task_map={c:task_for_class(c,class_order) for c in range(NUM_CLASSES)} if class_order is not None else None
    correct=0;invalid=0;perclass={i:{"correct":0,"total":0} for i in range(NUM_CLASSES)}
    task_matrix=[[0 for _ in range(TASKS)] for _ in range(TASKS)]
    for pred,gold in zip(predictions,gold_intent_ids):
        if not 0<=int(gold)<NUM_CLASSES: raise ValueError("gold class id outside range")
        parsed=label_to_id.get(normalize_prediction(pred));ok=parsed==int(gold);correct+=int(ok);invalid+=int(parsed is None)
        perclass[int(gold)]["total"]+=1;perclass[int(gold)]["correct"]+=int(ok)
        if task_map is not None and parsed is not None: task_matrix[task_map[int(gold)]][task_map[parsed]]+=1
    total=len(predictions)
    return {"metric":"free_generation_micro_exact_normalized_label_accuracy","correct":correct,"total":total,"accuracy":correct/total if total else 0.0,"invalid_predictions":invalid,"per_class":{str(k):{"correct":v["correct"],"total":v["total"],"accuracy":v["correct"]/v["total"] if v["total"] else None} for k,v in perclass.items()},"task_confusion":task_matrix if task_map is not None else None,"candidate_constrained":False}
