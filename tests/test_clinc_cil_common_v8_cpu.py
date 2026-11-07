import hashlib,json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"tools"))
import clinc_cil_common_v8 as c

class CharTokenizer:
    def encode(self,text,add_special_tokens=False): return [ord(x) for x in text]
    def decode(self,ids,skip_special_tokens=True): return ''.join(chr(i) for i in ids)

def hfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(files):
    classes=[f"intent_{i:03d}" for i in range(150)]
    return {"schema":c.SCHEMA,"classes":classes,"class_orders":{str(s):list(range(150)) for s in c.SEEDS},"files":files,"test_data_included":False,
        "source":{"name":"toy","metadata_sha256":"a"*64},
        "normalization":{"unicode":"NFKC","case":"casefold","strip":True,"underscore_space_equivalent":True}}

class ClincCILTests(unittest.TestCase):
    def test_natural_label_multitoken_ce_masks_prompt_and_padding(self):
        row={"id":"a"*64,"text_sha256":hashlib.sha256(b"hi").hexdigest(),"text":"hi","intent_id":4}
        tok=CharTokenizer();x=c.tokenize_row(row,tok,[f"intent_{i:03d}" for i in range(150)],eos_token_id=2)
        self.assertEqual(x["input_ids"][-len("intent_004")-1:-1],[ord(z) for z in "intent_004"])
        self.assertEqual(x["labels"][:x["prompt_length"]],[c.IGNORE_INDEX]*x["prompt_length"])
        self.assertEqual(x["labels"][-1],2);self.assertEqual(x["attention_mask"],[1]*len(x["input_ids"]))
        self.assertEqual(tok.decode(x["input_ids"][:-1]),c.build_prompt("hi")+"intent_004")
        batch=c.tokenize_batch([row,{"id":"b"*64,"text_sha256":hashlib.sha256(b"longer").hexdigest(),"text":"longer","intent_id":5}],tok,[f"intent_{i:03d}" for i in range(150)],2,0)
        self.assertEqual(batch["labels"][0][-1],c.IGNORE_INDEX);self.assertEqual(batch["attention_mask"][0][-1],0)
        clipped=c.tokenize_row(row,tok,[f"intent_{i:03d}" for i in range(150)],2,max_length=80)
        self.assertLessEqual(len(clipped["input_ids"]),80);self.assertEqual(clipped["labels"][-1],2)
        self.assertEqual(clipped["input_ids"][:len("Classify the intent of this user request.\nRequest: ")],[ord(z) for z in "Classify the intent of this user request.\nRequest: "])
        self.assertEqual(clipped["input_ids"][clipped["prompt_length"]-len("\nIntent:"):clipped["prompt_length"]],[ord(z) for z in "\nIntent:"])
        with self.assertRaises(ValueError):c.tokenize_row(row,tok,[f"intent_{i:03d}" for i in range(150)],2,prepend_bos=True)
        with_bos=c.tokenize_row(row,tok,[f"intent_{i:03d}" for i in range(150)],2,prepend_bos=True,bos_token_id=1)
        self.assertEqual(with_bos["input_ids"][0],1);self.assertEqual(with_bos["labels"][0],c.IGNORE_INDEX)
        long_text="request"*1000;very_long={**row,"text":long_text,"text_sha256":hashlib.sha256(long_text.encode()).hexdigest()}
        safe=c.tokenize_row(very_long,tok,[f"intent_{i:03d}" for i in range(150)],2,max_length=100,prepend_bos=True,bos_token_id=1)
        self.assertEqual(safe["input_ids"][0],1)
        self.assertTrue(safe["input_ids"][:len("Classify the intent of this user request.\nRequest: ")+1]==[1]+[ord(z) for z in "Classify the intent of this user request.\nRequest: "])
        self.assertEqual(safe["input_ids"][safe["prompt_length"]-len("\nIntent:"):safe["prompt_length"]],[ord(z) for z in "\nIntent:"])
        self.assertEqual(safe["labels"][-1],2)
        with self.assertRaises(ValueError):c.tokenize_row(row,tok,[f"intent_{i:03d}" for i in range(150)],2,max_length=25)
        with self.assertRaises(ValueError):c.tokenize_row({**row,"text":"changed"},tok,[f"intent_{i:03d}" for i in range(150)],2)

    def test_natural_names_normalize_and_invalid_counts_wrong(self):
        classes=[f"intent_{i:03d}" for i in range(150)]
        self.assertEqual(c.normalize_prediction("  INTENT  004\n"),"intent 004")
        r=c.score_predictions([" INTENT 004 ","not an intent"],[4,4],classes)
        self.assertEqual(r["correct"],1);self.assertEqual(r["total"],2);self.assertEqual(r["invalid_predictions"],1)
        self.assertAlmostEqual(r["accuracy"],.5);self.assertFalse(r["candidate_constrained"])

    def test_long_text_prompt_truncation_is_independent_of_gold_label(self):
        labels=sorted(["a","z"*40]+[f"intent_{i:03d}" for i in range(148)])
        short=labels.index("a");long=labels.index("z"*40)
        text="request sentence "*100
        rows=[{"id":"1"*64,"text_sha256":hashlib.sha256(text.encode()).hexdigest(),"text":text,"intent_id":short},
              {"id":"2"*64,"text_sha256":hashlib.sha256(text.encode()).hexdigest(),"text":text,"intent_id":long}]
        tok=CharTokenizer();p0=c.tokenize_prompt(text,tok,labels,2,max_length=100)
        a=c.tokenize_row(rows[0],tok,labels,2,max_length=100);b=c.tokenize_row(rows[1],tok,labels,2,max_length=100)
        self.assertEqual(a["input_ids"][:a["prompt_length"]],b["input_ids"][:b["prompt_length"]])
        self.assertEqual(a["input_ids"][:a["prompt_length"]],p0["input_ids"])
        self.assertLess(len(a["input_ids"]),len(b["input_ids"]))
        self.assertEqual(len(b["input_ids"]),100)

    def test_manifest_training_view_strips_test_and_streams_rows(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            root=Path(td);data={}
            for role,rows in {"train":[{"id":"a"*64,"text_sha256":hashlib.sha256(b"hello").hexdigest(),"text":"hello","intent_id":0}],"development":[{"id":"c"*64,"text_sha256":hashlib.sha256(b"dev").hexdigest(),"text":"dev","intent_id":1}],"test":[{"id":"e"*64,"text_sha256":hashlib.sha256(b"test").hexdigest(),"text":"test","intent_id":2}]}.items():
                p=root/f"{role}.jsonl";p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
                data[role]={"path":p.name,"sha256":hfile(p),"rows":len(rows)}
            m=manifest({k:v for k,v in data.items() if k!="test"});c.validate_manifest(m);v=c.make_training_view(m)
            self.assertFalse(hasattr(v,"test"));self.assertEqual(next(c.iter_role(v,"train",root))["intent_id"],0)
            with self.assertRaises(ValueError):c.resolve_role_path(v,"test",root)
            training_hash="1"*64;config_hash="2"*64;expected={(s,a) for s in (17,29,43) for a in ("HOPE","HOPE_Sleep")}
            tm={"schema":"clinc-cil-test-manifest-v8","classes":list(m["classes"]),"training_manifest_sha256":training_hash,"test":data["test"]}
            runs=[{"seed":s,"arm":a,"status":"complete","tasks_completed":10,"final_checkpoint_sha256":"3"*64,"run_receipt_sha256":"4"*64,"config_sha256":config_hash,"training_manifest_sha256":training_hash} for s,a in sorted(expected)]
            seal={"complete":True,"test_data_included":False,"training_manifest_sha256":training_hash,"config_sha256":config_hash,"runs":runs}
            kwargs={"expected_seed_arms":expected,"expected_config_sha256":config_hash,"expected_training_manifest_sha256":training_hash}
            with self.assertRaises(ValueError):list(c.iter_test_after_seal(tm,root,{**seal,"complete":False},**kwargs))
            self.assertEqual(len(list(c.iter_test_after_seal(tm,root,seal,**kwargs))),1)
            with self.assertRaises(ValueError):list(c.iter_test_after_seal(tm,root,{**seal,"runs":seal["runs"][:-1]},**kwargs))
            with self.assertRaises(ValueError):list(c.iter_test_after_seal(tm,root,{**seal,"config_sha256":"3"*64},**kwargs))
            bad=[dict(r) for r in runs];bad[0]["tasks_completed"]=9
            with self.assertRaises(ValueError):list(c.iter_test_after_seal(tm,root,{**seal,"runs":bad},**kwargs))
            bad=[dict(r) for r in runs];bad[0]["final_checkpoint_sha256"]="bad"
            with self.assertRaises(ValueError):list(c.iter_test_after_seal(tm,root,{**seal,"runs":bad},**kwargs))

    def test_task_partition_and_seeded_schedule(self):
        classes=[f"intent_{i:03d}" for i in range(150)]
        order=list(range(150));self.assertEqual(len(c.task_class_ids(order,9)),15)
        self.assertEqual(c.task_for_class(15,order),1)
        rows=[{"id":f"{i:064x}","intent_id":i%30} for i in range(60)]
        self.assertEqual(len(list(c.iter_task_rows(rows,order,0,classes))),30)
        self.assertEqual(c.shuffled_ids(rows,17,0),c.shuffled_ids(rows,17,0))
        self.assertNotEqual(c.shuffled_ids(rows,17,0),c.shuffled_ids(rows,29,0))

    def test_partition_catches_missing_class_and_cross_role_duplicates(self):
        train=[{"id":hashlib.sha256(f"t{cl}-{i}".encode()).hexdigest(),"intent_id":cl} for cl in range(150) for i in range(2)]
        dev=[{"id":hashlib.sha256(f"d{cl}-{i}".encode()).hexdigest(),"intent_id":cl} for cl in range(150) for i in range(1)]
        result=c.validate_role_partition(train,dev,train_per_class=2,development_per_class=1)
        self.assertEqual(result["rows_by_role"],{"train":300,"development":150})
        with self.assertRaises(ValueError):c.validate_role_partition(train,dev[:-1],train_per_class=2,development_per_class=1)
        with self.assertRaises(ValueError):c.validate_role_partition(train,train,train_per_class=2,development_per_class=2)

if __name__=="__main__":unittest.main()
