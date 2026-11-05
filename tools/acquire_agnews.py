"""Acquire pinned AG News bytes; test is hashed but never decoded here."""
import hashlib
import argparse
import json
import time
from pathlib import Path

import requests

OUT = Path('D:/my_paper/NL_Sleep_Public_Candidate/outputs/agnews-retention-v1/data')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', type=Path, default=OUT)
OUT = parser.parse_args().out_dir
REVISION = 'eb185aade064a813bc0b7f42de02595523103ca4'
FILES = {
    'train-00000-of-00001.parquet': (18585438, 'fc508d6d9868594e3da960a8cfeb63ab5a4746598b93428c224397080c1f52ee'),
    'test-00000-of-00001.parquet': (1234829, '71de87ec66bc5737752a2502204dfa6d7fe9856ade3ea444dc6317789a4f13fb'),
}
OUT.mkdir(parents=True, exist_ok=True)
evidence = {'dataset': 'fancyzhx/ag_news', 'revision': REVISION,
            'license_metadata': 'unknown',
            'source_card_note': 'Academic corpus supplied for research; no source text is redistributed.',
            'test_decoded': False, 'modelscope_probe': {}, 'files': {}}
session = requests.Session()
session.headers['User-Agent'] = 'NL-Sleep-research-dataset-acquisition/1'
try:
    metadata_url = 'https://modelscope.cn/api/v1/datasets/AI-ModelScope/ag_news'
    metadata_response = session.get(metadata_url, timeout=(10, 20))
    metadata = metadata_response.json()
    evidence['modelscope_probe'] = {'url': metadata_url, 'http_status': metadata_response.status_code,
                                    'code': metadata.get('Code'), 'message': metadata.get('Message')}
    dataset_id = metadata.get('Data', {}).get('Id') if isinstance(metadata.get('Data'), dict) else None
    if dataset_id:
        tree_url = f'https://modelscope.cn/api/v1/datasets/{dataset_id}/repo/tree'
        tree = session.get(tree_url, params={'Revision': 'master', 'Root': '/',
                           'Recursive': 'True', 'PageNumber': 1, 'PageSize': 100}, timeout=(10, 20))
        body = tree.json()
        evidence['modelscope_probe']['files'] = body.get('Data', {}).get('Files')
        evidence['modelscope_probe']['tree_http_status'] = tree.status_code
except Exception as exc:
    evidence['modelscope_probe']['error_type'] = type(exc).__name__
    evidence['modelscope_probe']['error'] = str(exc)[:400]
print(json.dumps({'modelscope_probe': evidence['modelscope_probe']}, ensure_ascii=False), flush=True)

# Exact public Parquet mirror is pinned by its immutable repository revision and
# LFS SHA256. Missing ModelScope dependencies/metadata do not change shared envs.
for filename, (expected_size, expected_sha) in FILES.items():
    destination = OUT / filename
    url = f'https://huggingface.co/datasets/fancyzhx/ag_news/resolve/{REVISION}/data/{filename}?download=true'
    if destination.exists():
        data_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
        if destination.stat().st_size != expected_size or data_sha != expected_sha:
            raise RuntimeError('existing dataset byte mismatch: ' + filename)
    else:
        partial = OUT / (filename + f'.partial-{time.time_ns()}')
        digest = hashlib.sha256()
        count = 0
        with session.get(url, stream=True, timeout=(15, 60)) as response:
            response.raise_for_status()
            with partial.open('xb') as stream:
                for block in response.iter_content(1 << 20):
                    if not block:
                        continue
                    count += len(block)
                    if count > expected_size:
                        raise RuntimeError('download exceeds pinned size: ' + filename)
                    digest.update(block)
                    stream.write(block)
        data_sha = digest.hexdigest()
        if count != expected_size or data_sha != expected_sha:
            raise RuntimeError('download content mismatch: ' + filename)
        partial.rename(destination)
    evidence['files'][filename] = {'bytes': expected_size, 'sha256': expected_sha,
                                  'url': url, 'decoded': False}
    print(json.dumps({'verified': filename, 'bytes': expected_size, 'sha256': expected_sha}), flush=True)
(OUT / 'acquisition.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PINNED_BYTES_ACQUIRED_TEST_NOT_DECODED', 'directory': str(OUT)}), flush=True)
