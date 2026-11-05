"""Prepare only AG News TRAIN-derived roles. Official test is never opened."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import unicodedata

DATA = Path('D:/my_paper/NL_Sleep_Public_Candidate/outputs/agnews-retention-v1/data')
MODEL = Path('D:/my_paper/Nested_Learning_and_Sleep/.cache/models/llama32-3b')
CREATED_UTC = None
OUTPUT_FILE = None
TRAIN_SHA = 'fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee'
REVISION = 'eb185aade064a813bc0b7f42de02595523103ca4'
PREFIX = 'Read the news article and return its assigned code as tag plus one word.\nArticle:\n'
SUFFIX = '\nAnswer:'
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
OLD, NEW = [0, 1], [2, 3]
SEEDS = (1907, 17, 29, 43)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'))
        stream.write('\n')


def select_candidates():
    import pyarrow.parquet as pq
    source = DATA / 'train-00000-of-00001.parquet'
    assert digest(source) == TRAIN_SHA
    table = pq.read_table(source, columns=['text', 'label'])
    assert len(table) == 120000
    groups = defaultdict(list)
    for index, row in enumerate(table.to_pylist()):
        text, label = row['text'], int(row['label'])
        assert isinstance(text, str) and label in range(4)
        normalized = ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()
        identifier = hashlib.sha256(normalized.encode()).hexdigest()
        groups[identifier].append({'record_id_sha256': identifier, 'source_index': index,
                                   'class_id': label, 'text': text})
    buckets = defaultdict(list)
    conflicts = 0
    duplicates = 0
    for identifier, rows in groups.items():
        duplicates += len(rows)-1
        if len({r['class_id'] for r in rows}) != 1:
            conflicts += 1
            continue
        row = min(rows, key=lambda r: r['source_index'])
        row['selection_key'] = hashlib.sha256(('nl-sleep-agnews-v1\0'+identifier).encode()).hexdigest()
        buckets[row['class_id']].append(row)
    selected = []
    for label in range(4):
        candidates = sorted(buckets[label], key=lambda r: r['selection_key'])[:2048]
        assert len(candidates) == 2048
        selected.extend(candidates)
    result = {'schema': 'agnews-train-candidates-private-v1', 'source_train_sha256': TRAIN_SHA,
              'rows_read': len(table), 'normalized_duplicate_rows': duplicates,
              'conflicting_normalized_text_groups_dropped': conflicts,
              'candidate_count_each_class': 2048, 'test_opened': False, 'records': selected}
    write_new(DATA / 'train-candidates-private.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


def tokenize_records():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    source_candidates = DATA / 'train-candidates-private.json'
    raw = json.loads(source_candidates.read_text(encoding='utf-8'))
    assert raw['source_train_sha256'] == TRAIN_SHA and raw['test_opened'] is False
    prefix = tokenizer.encode(PREFIX, add_special_tokens=True)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    answers = [tokenizer.encode(' tag'+word, add_special_tokens=False)
               for word in (' red', ' blue', ' green', ' gold')]
    assert all(len(a)==2 for a in answers)
    assert len({a[0] for a in answers})==1 and len({a[1] for a in answers})==4
    for word, answer in zip((' red', ' blue', ' green', ' gold'), answers):
        assert tokenizer.encode(SUFFIX+' tag'+word, add_special_tokens=False)==suffix+answer
    assert len(prefix)+96+len(suffix)<=128
    by_prompt = defaultdict(list)
    for row in raw['records']:
        article = tokenizer.encode(row['text'], add_special_tokens=False)
        assert article
        tokens = prefix+article[:96]+suffix
        by_prompt[object_sha(tokens)].append({
            **{k:row[k] for k in ('record_id_sha256','source_index','class_id','selection_key')},
            'tokens':tokens, 'prompt_sha256':object_sha(tokens),
            'article_token_count':min(len(article),96), 'article_was_truncated':len(article)>96})
    buckets = defaultdict(list)
    conflicts = 0
    for rows in by_prompt.values():
        if len({r['class_id'] for r in rows})!=1:
            conflicts += 1
            continue
        row = min(rows, key=lambda r:r['selection_key'])
        buckets[row['class_id']].append(row)
    records = []
    layout = [('dev','train',128),('dev','calib',64),('main','train',256),('main','calib',64)]
    for label in range(4):
        ordered = sorted(buckets[label], key=lambda r:r['selection_key'])
        cursor = 0
        for partition, role, count in layout:
            selected = ordered[cursor:cursor+count]
            assert len(selected)==count
            cursor += count
            for row in selected:
                row = dict(row)
                row.pop('selection_key')
                phase = 'old' if label in OLD else 'new'
                records.append({**row, 'partition':partition, 'phase':phase, 'role':phase+'_'+role})
    assert len({r['record_id_sha256'] for r in records})==len(records)
    assert len({r['prompt_sha256'] for r in records})==len(records)
    index_sets = {}
    for partition in ('dev','main'):
        for label in range(4):
            index_sets[(partition,label)] = [i for i,r in enumerate(records)
                if r['partition']==partition and r['class_id']==label and r['role'].endswith('_train')]
    schedules, code_maps, pools = {}, {}, {}
    for seed in SEEDS:
        partition = 'dev' if seed==1907 else 'main'
        palette = [a[1] for a in answers]
        random.Random(seed+870000).shuffle(palette)
        code_maps[str(seed)] = {'marker_token_id':answers[0][0], 'class_to_token_ids':palette}
        schedules[str(seed)] = {}
        for name, labels, count, offset in (('foundation512',OLD,512,1),('new_primary256',NEW,256,2)):
            rng = random.Random(seed*10000+offset)
            batches = []
            for _ in range(count):
                batch = [rng.choice(index_sets[(partition,label)]) for label in labels for _ in range(2)]
                rng.shuffle(batch)
                batches.append(batch)
            schedules[str(seed)][name] = batches
            schedules[str(seed)][name+'_sha256'] = object_sha(batches)
        pools[str(seed)] = []
        class_sequence = [0,2,3,1,1,3,2,0]*2+[0,2,1,3]*2
        for event in (64,128,192,256):
            rng = random.Random(seed*10000+event+3000)
            pool = [rng.choice(index_sets[(partition,label)]) for label in class_sequence]
            assert Counter(records[i]['class_id'] for i in pool)=={0:6,1:6,2:6,3:6}
            pools[str(seed)].append(pool)
    inventory = [{ 'name':p.name,'bytes':p.stat().st_size,'sha256':digest(p)}
        for p in sorted(MODEL.iterdir()) if p.is_file() and p.suffix in ('.json','.txt')]
    counts = Counter(f"{r['partition']}/{r['role']}/class{r['class_id']}" for r in records)
    result = {'schema':'nl-sleep-agnews-train-roles-v1', 'created_utc':CREATED_UTC or datetime.now(timezone.utc).isoformat(),
       'source':{'provider':'HuggingFace fancyzhx/ag_news; ModelScope metadata returned404',
                 'revision':REVISION,'split':'train','source_file_sha256':TRAIN_SHA,
                 'license_metadata':'unknown; source corpus research use; raw data not redistributed'},
       'class_names':CLASS_NAMES,'phase_class_ids':{'old':OLD,'new':NEW},
       'tokenizer':{'source':'existing ModelScope Llama3.2-3B assets','file_inventory':inventory,
                    'pad_token_id':tokenizer.eos_token_id},
       'prompt':{'prefix':PREFIX,'answer_cue':SUFFIX,'prompt_prefix_token_ids':prefix,
                 'answer_cue_token_ids':suffix,'article_token_cap':96,'max_prompt_tokens':128,
                 'composition':'encode prefix with special tokens, append first96 article tokens (no specials), then suffix tokens (no specials)'},
       'selection':{'salt':'nl-sleep-agnews-v1','normalized_train_conflicts_dropped':raw['conflicting_normalized_text_groups_dropped'],
                    'prompt_conflict_groups_dropped':conflicts,'selected_counts':dict(counts),
                    'rule':'NFKC+whitespace+casefold text dedup; hash order, token-prefix dedup; disjoint dev/main train/calib articles'},
       'test_opened':False,'records':records,'code_maps':code_maps,'schedules':schedules,
       'sleep_prefix_pools':pools}
    path = OUTPUT_FILE or DATA/'agnews-train-roles-v1.json'
    write_new(path,result)
    print(json.dumps({'status':'TRAIN_ROLES_FROZEN_TEST_UNOPENED','file':str(path),'sha256':digest(path),
                      'record_count':len(records),'prompt_length_min':min(len(r['tokens'])for r in records),
                      'prompt_length_max':max(len(r['tokens'])for r in records),'selected_counts':dict(counts),
                      'prefix_tokens':len(prefix),'cue_tokens':len(suffix),'test_opened':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['select','tokenize'], required=True)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--model-dir', type=Path, default=MODEL)
    parser.add_argument('--created-utc', help='preserve the recorded timestamp for byte-exact regeneration')
    parser.add_argument('--output-file', type=Path, help='tokenize output; must not already exist')
    args=parser.parse_args()
    DATA, MODEL, CREATED_UTC, OUTPUT_FILE = args.data_dir, args.model_dir, args.created_utc, args.output_file
    select_candidates() if args.stage=='select' else tokenize_records()
