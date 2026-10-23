"""Parameter checks only. This command never imports transformers or loads assets."""
import argparse
import json
from pathlib import Path
from .execution import select_executor

def validate(local_model, *, rank=4, capacity=8, levels=2, executor='REF'):
    if not local_model:
        raise ValueError('an explicit existing local model directory is required; no download fallback')
    path = Path(local_model)
    if not path.is_dir() or path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
        raise ValueError('model path must be a real local directory')
    if not (path / 'config.json').is_file():
        raise ValueError('local config.json is required')
    cfg = json.loads((path / 'config.json').read_text(encoding='utf-8'))
    if cfg.get('model_type') != 'llama' or not isinstance(cfg.get('hidden_size'), int):
        raise ValueError('expected Llama model metadata')
    if rank < 1 or capacity < 1 or levels != 2 or rank > cfg['hidden_size']:
        raise ValueError('expected positive rank/capacity and exactly two CMS levels')
    decision = select_executor(executor)
    return {'status': 'NOT_RUN', 'check': 'metadata and parameters only',
            'executor': decision['executor'], 'rank': rank, 'capacity': capacity,
            'levels': levels, 'insertion': 'last MLP output', 'backbone': 'frozen by caller',
            'assets_loaded': False}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--rank', type=int, default=4)
    p.add_argument('--capacity', type=int, default=8)
    p.add_argument('--levels', type=int, default=2)
    p.add_argument('--executor', default='REF')
    a = p.parse_args()
    try:
        print(json.dumps(validate(a.model_path,rank=a.rank,capacity=a.capacity,levels=a.levels,executor=a.executor)))
    except (ValueError, RuntimeError) as exc:
        p.error(str(exc))
if __name__ == '__main__':
    main()
