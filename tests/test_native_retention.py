import importlib.util
import json
from pathlib import Path
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'tools/native_retention.py').exists())
spec = importlib.util.spec_from_file_location('native_retention_runner', ROOT/'tools/native_retention.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_synthetic_split_is_disjoint_balanced_and_target_does_not_use_noise():
    for task in ('old_a', 'new_b', 'copy'):
        train, evaluation = runner.pool(task), runner.pool(task, evaluation=True)
        assert len(train) == len(evaluation) == 512
        assert not {tuple(x) for x, _ in train} & {tuple(x) for x, _ in evaluation}
        assert all(sum(y == 32+k for _, y in evaluation) == 128 for k in range(4))
    assert runner.pool('old_a')[0][1] == 32
    assert runner.pool('new_b')[0][1] == 33


def test_correctness_fixtures_cover_stream_state_and_period_one_updates():
    cfg = json.loads((ROOT/'configs/native-retention-v1.json').read_text())
    torch.set_num_threads(1)
    result = runner.correctness(cfg)
    assert result['stream_max_abs_error'] == 0
    assert result['period_one_matches_sft']


def test_main_budget_matches_three_seed_fixed_schedule():
    cfg = json.loads((ROOT/'configs/native-retention-v1.json').read_text())
    assert len(cfg['seeds']) * (cfg['foundation_steps'] + 2*cfg['adapt_steps']) == cfg['max_update_attempts']
    assert all(cfg['adapt_steps'] % period == 0 for period in cfg['cms_periods'])
