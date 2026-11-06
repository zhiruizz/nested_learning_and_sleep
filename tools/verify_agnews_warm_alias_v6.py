"""CPU-only provenance bridge from qualified warm files to v5 step-zero copies.

File hashes need not match: training checkpoint metadata differs. This checks
the full adapter tensor state and execution policy actually used for scoring.
No data pack, model forward, optimizer or GPU is used.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(warm_dir: Path, v5_dir: Path, initial_protocol: Path) -> dict:
    warm_dir, v5_dir = warm_dir.resolve(), v5_dir.resolve()
    protocol = json.loads(initial_protocol.read_text(encoding='utf-8'))
    warm_result = warm_dir / 'transfer-result.json'
    seal_path = v5_dir / 'development-sealed.json'
    assert sha(warm_result) == protocol['source_sha256']['warm_result_sha256']
    assert sha(seal_path) == protocol['source_sha256']['v5_seal_sha256']
    warm = json.loads(warm_result.read_text(encoding='utf-8'))
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    rows = []
    for seed in (17, 29, 43):
        original = next(r['checkpoint'] for r in warm['qualified_models'] if r['seed'] == seed)
        run = next(r for r in seal['runs'] if r['seed'] == seed and r['arm'] == 'warm_CMS_reference')
        copied = next(r for r in run['checkpoints'] if r['primary_step'] == 0)
        paths = [(base / row['path']).resolve() for base, row in ((warm_dir, original), (v5_dir, copied))]
        for base, record, path in zip((warm_dir, v5_dir), (original, copied), paths):
            assert path.is_relative_to(base)
            assert sha(path) == record['sha256']
        assert original['sha256'] == protocol['source_sha256']['checkpoint_sha256_by_seed'][str(seed)]['warm']
        a, b = [torch.load(p, map_location='cpu', weights_only=False) for p in paths]
        assert a['model'].keys() == b['model'].keys()
        assert all(torch.equal(a['model'][k], b['model'][k]) for k in a['model'])
        assert a['execution_state'] == b['execution_state']
        base = {k: int(torch.count_nonzero(t)) for k, t in a['model'].items()
                if '.mlp.memory.' in k and '.experts.' not in k}
        active = sum(int(torch.count_nonzero(t)) for k, t in a['model'].items()
                     if '.mlp.memory.' in k and k.endswith('.active'))
        rows.append({'seed': seed, 'qualified_warm_sha256': original['sha256'],
                     'v5_step0_sha256': copied['sha256'], 'qualified_warm_file_bytes': paths[0].stat().st_size,
                     'v5_step0_file_bytes': paths[1].stat().st_size,
                     'all_model_tensors_exactly_equal': True, 'execution_policy_exactly_equal': True,
                     'model_tensor_count': len(a['model']), 'warm_base_nonzero_counts': base,
                     'warm_active_expert_count': active})
    return {'schema': 'agnews-warm-alias-provenance-v6-v1', 'passed': True, 'rows': rows,
            'model_forwards': 0, 'optimizer_updates': 0, 'source_sha256': sha(Path(__file__)),
            'initial_protocol_sha256': sha(initial_protocol),
            'scope': 'Exact inference adapter state and policy, not byte-identical checkpoint files or training controller/RNG.'}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--warm-dir', type=Path, required=True)
    p.add_argument('--v5-run-dir', type=Path, required=True)
    p.add_argument('--initial-protocol', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    args = p.parse_args()
    assert not args.out_json.exists()
    result = verify(args.warm_dir, args.v5_run_dir, args.initial_protocol)
    args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'passed': True, 'seeds': 3, 'model_forwards': 0,
                      'warm_file_bytes': [r['qualified_warm_file_bytes'] for r in result['rows']]}))


if __name__ == '__main__':
    main()
