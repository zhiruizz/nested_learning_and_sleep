"""Bounded synthetic retention pilot; see docs/native-retention-protocol.md."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import torch
from torch.nn import functional as F
from nested_sleep.accounting import WorkMeter, ACTIVE_METER
from nested_sleep.lm import NativeResearchLM
from nested_sleep.native import NativeConfig
from nested_sleep.optim import CMSController
from nested_sleep.parameter_drift import parameter_drift


def pool(task, *, evaluation=False):
    rows = []
    for a in range(16):
        for b in range(16):
            if (a + b) % 2 != int(evaluation):
                continue
            for key in range(4):
                if task == 'copy':
                    x, y = [42, 32 + key, a, b], 32 + key
                elif task in ('old_a', 'new_b'):
                    x = [a, b, 40 if task == 'old_a' else 41, 36 + key]
                    y = 32 + (key if task == 'old_a' else (key + 1) % 4)
                else:
                    raise ValueError(task)
                rows.append((x, y))
    return rows


def tensors(rows):
    return (torch.tensor([r[0] for r in rows], dtype=torch.long),
            torch.tensor([r[1] for r in rows], dtype=torch.long))


def sample(rows, rng, count):
    return tensors([rows[rng.randrange(len(rows))] for _ in range(count)])


def snapshots(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}


def digest_model(model):
    h = hashlib.sha256()
    for n, p in sorted(model.state_dict().items()):
        h.update(n.encode())
        h.update(p.cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_model(cfg, seed):
    torch.manual_seed(seed)
    model = NativeResearchLM(NativeConfig(**cfg['model']), cfg['rank'], cfg['capacity'])
    model.banks()[0].activate(0)
    return model


@torch.no_grad()
def evaluate(model):
    model.eval()
    before = digest_model(model)
    result = {}
    for task in ('old_a', 'copy', 'new_b'):
        rows = pool(task, evaluation=True)
        nll, greedy, candidate = 0., 0, 0
        for i in range(0, len(rows), 64):
            x, y = tensors(rows[i:i+64])
            logits = model.logits(x)[:, -1].double()
            nll += F.cross_entropy(logits, y, reduction='sum').item()
            greedy += int((logits.argmax(-1) == y).sum())
            candidate += int((logits[:, 32:36].argmax(-1) + 32 == y).sum())
        result[task] = {'count': len(rows), 'greedy_correct': greedy,
                        'candidate_correct': candidate, 'accuracy': greedy / len(rows),
                        'candidate_accuracy': candidate / len(rows),
                        'nll': nll / len(rows), 'answer_ppl': math.exp(nll / len(rows))}
    if digest_model(model) != before:
        raise RuntimeError('evaluation changed model state')
    return result


def correctness(cfg):
    model = make_model(cfg, 907).eval()
    x, _ = tensors(pool('old_a')[:8])
    with torch.no_grad():
        full = model.logits(x)
        state, pieces = None, []
        for i in range(x.shape[1]):
            logits, state = model.core(x[:, i:i+1], state)
            pieces.append(logits)
        error = float((full - torch.cat(pieces, 1)).abs().max())
        torch.testing.assert_close(full, torch.cat(pieces, 1), rtol=0, atol=0)
    left, right = deepcopy(model), deepcopy(model)
    c1 = CMSController(left, (1, 1), cfg['adapt_lr'], cfg['adapt_lr'])
    c2 = CMSController(right, (1, 1), cfg['adapt_lr'], cfg['adapt_lr'])
    for m, c, sequential in ((left, c1, True), (right, c2, False)):
        c.prepare_wake(m)
        x, y = tensors(pool('new_b')[:8])
        F.cross_entropy(m.logits(x)[:, -1], y).backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg['clip_norm'])
        proposals = c.collect(m, sequential=sequential)
        for i, proposal in sorted(proposals.items()):
            c.apply(m, i, proposal)
    if digest_model(left) != digest_model(right):
        raise RuntimeError('period-one CMS differs from sequential update')
    for task in ('old_a', 'new_b', 'copy'):
        a, b = pool(task), pool(task, evaluation=True)
        assert len(a) == len(b) == 512
        assert not {tuple(x) for x, _ in a} & {tuple(x) for x, _ in b}
        assert all(sum(y == 32+k for _, y in b) == 128 for k in range(4))
    return {'stream_max_abs_error': error, 'period_one_matches_sft': True,
            'fixture_update_attempts': 2, 'split_disjoint_and_labels_balanced': True}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run(cfg, output, config_bytes):
    if cfg['device'] != 'cpu':
        raise ValueError('v1 is pinned to CPU')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    ledger = {'status': 'running', 'update_attempts': 0, 'completed_updates': 0}
    meter = WorkMeter()
    token = ACTIVE_METER.set(meter)
    code_root = Path(__file__).resolve().parents[1]
    provenance = {'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                  'python': platform.python_version(), 'torch': torch.__version__,
                  'platform': platform.platform(), 'device': 'cpu',
                  'git_commit': subprocess.check_output(['git', '-C', str(code_root), 'rev-parse', 'HEAD'], text=True).strip(),
                  'source_sha256': {}, 'evaluation_sources': 'new synthetic only',
                  'wall_seconds_limit': cfg['wall_seconds']}
    import inspect
    import nested_sleep.lm, nested_sleep.native, nested_sleep.optim, nested_sleep.experts
    for module in (nested_sleep.lm, nested_sleep.native, nested_sleep.optim, nested_sleep.experts):
        path = Path(inspect.getfile(module))
        provenance['source_sha256'][module.__name__] = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance['source_sha256']['runner'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(output/'provenance.json', provenance)
    write(output/'config.json', cfg)

    def charge_attempt():
        if ledger['update_attempts'] >= cfg['max_update_attempts']:
            raise RuntimeError('update cap reached')
        if time.monotonic() - started >= cfg['wall_seconds']:
            raise RuntimeError('wall cap reached')
        ledger['update_attempts'] += 1
        write(output/'ledger.json', ledger)

    rows, drift, scopes = [], [], []
    try:
        meter.stage = 'correctness'
        ledger['fixture_attempts_reserved'] = 2
        write(output/'ledger.json', ledger)
        write(output/'correctness.json', correctness(cfg))
        ledger['fixture_completed_updates'] = 2
        ledger['fixture_core_stream_tokens_outside_work_meter'] = 32
        for seed in cfg['seeds']:
            model = make_model(cfg, seed)
            print('seed', seed, 'foundation start', flush=True)
            meter.stage = 'evaluation'
            rows.append({'seed': seed, 'arm': 'random', 'step': 0, 'metrics': evaluate(model)})
            initial = snapshots(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['foundation_lr'], weight_decay=0.)
            rng = random.Random(seed + 10000)
            old, copy = pool('old_a'), pool('copy')
            meter.stage = 'foundation'
            model.train()
            foundation_start = time.monotonic()
            for step in range(cfg['foundation_steps']):
                charge_attempt()
                x, y = sample(old if step % 2 == 0 else copy, rng, cfg['batch_size'])
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model.logits(x)[:, -1], y)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite foundation loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                ledger['completed_updates'] += 1
            foundation_seconds = time.monotonic() - foundation_start
            meter.stage = 'evaluation'
            foundation = evaluate(model)
            rows.append({'seed': seed, 'arm': 'foundation', 'step': cfg['foundation_steps'], 'metrics': foundation,
                         'seconds': foundation_seconds})
            drift.append({'seed': seed, 'stage': 'foundation', 'parameters': parameter_drift(initial, snapshots(model))})
            base_hash = digest_model(model)
            rows.append({'seed': seed, 'arm': 'frozen', 'step': cfg['adapt_steps'], 'metrics': foundation,
                         'seconds': 0., 'model_sha256': base_hash, 'reused_identical_foundation_metrics': True})
            for arm in ('sft', 'cms'):
                candidate = deepcopy(model)
                ctrl = CMSController(candidate, cfg['cms_periods'], cfg['adapt_lr'], cfg['adapt_lr'])
                ctrl.prepare_wake(candidate)
                scope = [n for n, p in candidate.named_parameters() if p.requires_grad]
                scopes.append({'seed': seed, 'arm': arm, 'names': scope,
                               'parameter_count': sum(p.numel() for p in candidate.parameters() if p.requires_grad),
                               'total_parameters': sum(p.numel() for p in candidate.parameters())})
                before = snapshots(candidate)
                source_hash = digest_model(candidate)
                if source_hash != base_hash:
                    raise RuntimeError('arm starting state mismatch')
                rng = random.Random(seed + 20000)
                data_hash = hashlib.sha256()
                train = pool('new_b')
                writes = [0, 0]
                train_seconds = 0.
                for step in range(1, cfg['adapt_steps'] + 1):
                    meter.stage = 'adapt_' + arm
                    charge_attempt()
                    t = time.monotonic()
                    candidate.train()
                    ctrl.prepare_wake(candidate)
                    x, y = sample(train, rng, cfg['batch_size'])
                    data_hash.update(x.numpy().tobytes()); data_hash.update(y.numpy().tobytes())
                    loss = F.cross_entropy(candidate.logits(x)[:, -1], y)
                    if not torch.isfinite(loss):
                        raise RuntimeError('nonfinite adaptation loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in candidate.parameters() if p.requires_grad], cfg['clip_norm'], error_if_nonfinite=True)
                    proposals = ctrl.collect(candidate, sequential=(arm == 'sft'))
                    for i, proposal in sorted(proposals.items()):
                        ctrl.apply(candidate, i, proposal)
                        writes[i] += 1
                    ledger['completed_updates'] += 1
                    train_seconds += time.monotonic() - t
                    if step in cfg['eval_steps']:
                        meter.stage = 'evaluation'
                        rows.append({'seed': seed, 'arm': arm, 'step': step, 'metrics': evaluate(candidate),
                                     'seconds': train_seconds, 'base_writes': writes.copy(),
                                     'data_sha256': data_hash.hexdigest(), 'start_model_sha256': source_hash,
                                     'pending_counts': ctrl.counts.copy()})
                changed = parameter_drift(before, snapshots(candidate))
                if any(r['changed'] for r in changed if r['name'] not in scope):
                    raise RuntimeError('frozen parameter changed')
                drift.append({'seed': seed, 'stage': arm, 'parameters': changed})
                write(output/'metrics.json', rows)
                write(output/'parameter-drift.json', drift)
                write(output/'scopes.json', scopes)
                print('seed', seed, arm, rows[-1]['metrics'], flush=True)
            a = [r for r in rows if r['seed'] == seed and r['arm'] == 'sft' and r['step'] == cfg['adapt_steps']][0]
            b = [r for r in rows if r['seed'] == seed and r['arm'] == 'cms' and r['step'] == cfg['adapt_steps']][0]
            assert a['data_sha256'] == b['data_sha256']
        ledger['status'] = 'completed'
    except BaseException as exc:
        ledger['status'] = 'failed_or_interrupted'
        ledger['error'] = repr(exc)
        raise
    finally:
        ledger['elapsed_seconds'] = time.monotonic() - started
        ledger['work_meter'] = meter.counts
        ledger['supervised_answer_tokens'] = ledger['completed_updates'] * cfg['batch_size']
        ledger['fixture_supervised_answer_tokens'] = 16 if ledger.get('fixture_completed_updates') == 2 else None
        write(output/'ledger.json', ledger)
        write(output/'metrics.json', rows)
        ACTIVE_METER.reset(token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    run(json.loads(raw), args.output, raw)
