"""Verify AG News v4 aggregate/source hashes against the actual Git index.

Reads staged public artifacts only. Does not load model checkpoints, articles,
prompts, labels, Torch, or CUDA. Run at the public repository root after staging.
"""
import hashlib
import json
import math
import subprocess

BASE = 'results/agnews-preupdate-v4/'
SEEDS = (17, 29, 43)
EM = 'full_vocab_two_token_greedy_exact_match'
STEPS = [0, 16, 32, 64, 112]


def blob(path):
    return subprocess.check_output(['git', 'show', ':' + path])


def sha(path):
    return hashlib.sha256(blob(path)).hexdigest()


def read(path):
    return json.loads(blob(path))


def verify():
    cfg_path = 'configs/agnews-boundary-transfer-v4.json'
    cfg = read(cfg_path)
    warm = read(BASE + 'transfer-result.json')
    wp = read(BASE + 'warm-protocol.json')
    lock = read(BASE + 'phaseB-source-lock.json')
    teacher = read(BASE + 'teacher-bank-manifest.json')
    protocol = read(BASE + 'protocol.json')
    seal = read(BASE + 'development-sealed.json')
    analysis = read(BASE + 'analysis.json')
    reserve = read(BASE + 'confirmation-reservation-manifest.json')
    assert warm['config_sha256'] == wp['config_sha256'] == lock['config_sha256'] == seal['config_sha256'] == sha(cfg_path)
    assert wp['config'] == protocol['config'] == cfg
    assert warm['source_sha256'] == cfg['warm_source_sha256'] == sha('tools/qualify_agnews_boundary_transfer_v4.py')
    assert warm['protocol_sha256'] == sha(BASE + 'warm-protocol.json')
    assert cfg['boundary_helper_sha256'] == lock['boundary_helper_sha256'] == sha('tools/agnews_preupdate_sleep_v4.py')
    assert lock['driver_source_sha256'] == seal['driver_source_sha256'] == sha('tools/run_agnews_preupdate_comparison_v4.py')
    assert lock['warm_result_sha256'] == seal['warm_result_sha256'] == protocol['warm_result_sha256'] == sha(BASE + 'transfer-result.json')
    assert lock['old_teacher_bank_manifest_sha256'] == sha(BASE + 'teacher-bank-manifest.json')
    assert protocol['source_lock'] == lock
    assert seal['source_lock_sha256'] == protocol['source_lock_sha256'] == sha(BASE + 'phaseB-source-lock.json')
    assert seal['protocol_sha256'] == sha(BASE + 'protocol.json')
    assert analysis['input_hashes'] == {
        'warm_result_sha256': sha(BASE + 'transfer-result.json'),
        'development_seal_sha256': sha(BASE + 'development-sealed.json'),
        'protocol_sha256': sha(BASE + 'protocol.json')}
    assert cfg['confirmation_reservation_manifest_sha256'] == sha(BASE + 'confirmation-reservation-manifest.json')
    assert reserve['status'] == 'reserved_unscored' and reserve['scored'] is False
    assert reserve['counts']['reserved_total'] == 1024
    assert all(reserve['intersections'].values())
    assert cfg['v3_main_seal_sha256'] == sha('results/agnews-qualified-v3/training-sealed.json')
    for module, expected in cfg['nested_source_sha256'].items():
        p = ('src/nested_sleep/__init__.py' if module == 'nested_sleep'
             else 'src/' + module.replace('.', '/') + '.py')
        assert sha(p) == expected, p

    teachers = {int(x['seed']): x for x in teacher['teachers']}
    assert set(teachers) == set(SEEDS)
    assert warm['all3_qualified'] and warm['phaseB_permitted']
    assert warm['confirmation_scored'] is False
    assert [s['budget'] for s in warm['stages']] == [24, 96, 384, 1536]
    first_common = None
    for stage in warm['stages']:
        assert {int(x['seed']) for x in stage['records']} == set(SEEDS)
        flags = []
        for row in stage['records']:
            t = warm['teachers'][str(row['seed'])]['validation']
            va, tr = row['validation'], row['train']
            gate = (tr['old'][EM] >= .8 and va['old'][EM] >= .75
                    and all(va[g][EM] >= .7 for g in ('class0', 'class1'))
                    and all(va[g][EM] >= t[g][EM] - .05 - 1e-12
                            for g in ('old', 'class0', 'class1')))
            assert gate == row['qualified'], (stage['budget'], row['seed'])
            flags.append(gate)
        assert all(flags) == stage['all_qualified']
        if all(flags) and first_common is None:
            first_common = stage['budget']
    assert first_common == warm['chosen_common_budget'] == lock['chosen_shared_warm_budget'] == 1536
    assert warm['successful_or_attempted_main_Adam_updates'] == 4608 and warm['extra_ghost_Adam_updates'] == 2
    assert warm['restore_ghost_test']['two_independent_restores_identical']
    assert warm['restore_ghost_test']['saved_snapshot_not_mutated']
    qualified = {int(x['seed']): x for x in warm['qualified_models']}
    last = {int(x['seed']): x for x in warm['stages'][-1]['records']}
    for seed in SEEDS:
        assert teachers[seed]['checkpoint_sha256'] == warm['teachers'][str(seed)]['foundation_sha256']
        assert qualified[seed]['warm_updates'] == 1536
        assert qualified[seed]['train'] == last[seed]['train']
        assert qualified[seed]['validation'] == last[seed]['validation']
        assert qualified[seed]['checkpoint']['adapter_fingerprint'] == last[seed]['student_fingerprint']

    assert seal['complete'] and seal['exploratory_only'] and seal['no_holdout_input']
    assert seal['confirmation_scored'] is False
    assert seal['actual_supervised_updates'] == 3024 and seal['actual_Sleep_Adam_updates'] == 4104
    runs = seal['runs']
    assert len(runs) == 18
    assert {(x['seed'], x['arm']) for x in runs} == {(s, a) for s in SEEDS for a in cfg['phaseB_arms']}
    common_start = {}
    common_policy = {}
    for run in runs:
        seed, arm = run['seed'], run['arm']
        assert run['status'] == 'complete' and run['primary_updates'] == 112
        assert [x['primary_step'] for x in run['validation']] == STEPS
        assert [x['primary_step'] for x in run['checkpoints']] == STEPS
        expected_events = 28 if arm in ('warm_pre_sleep_P4', 'warm_post_sleep_P4') else 1 if arm == 'warm_pre_sleep_P64' else 0
        assert len(run['events']) == expected_events
        assert run['sleep_Adam_updates'] == expected_events * 24
        assert run['extra_replay_updates'] == (336 if arm == 'warm_extra_gold_replay_P4' else 0)
        initial = run['validation'][0]
        if arm != 'foundation_CMS_reference':
            assert initial['adapter_fingerprint'] == qualified[seed]['checkpoint']['adapter_fingerprint']
            assert initial['metrics']['old'] == qualified[seed]['validation']['old']
            if seed in common_start:
                assert initial['metrics'] == common_start[seed]
                assert initial['execution_policy'] == common_policy[seed]
            common_start[seed] = initial['metrics']
            common_policy[seed] = initial['execution_policy']
        else:
            assert initial['metrics']['old'] == warm['teachers'][str(seed)]['validation']['old']
        for v in run['validation']:
            m = v['metrics']
            for group, n in [('old', 256), ('new', 256), ('all', 512),
                             ('class0', 128), ('class1', 128), ('class2', 128), ('class3', 128)]:
                x = m[group]
                assert x['examples'] == n and 0 <= x[EM] <= 1
                assert abs(x[EM] * n - round(x[EM] * n)) < 1e-8
                assert math.isfinite(x['answer_nll_nats_per_token'])
            assert m['all'][EM] == (m['old'][EM] + m['new'][EM]) / 2
        seen_slots = []
        for event_index, event in enumerate(run['events']):
            assert event['total_adam_updates'] == 24 and event['candidate_committed']
            assert event['teacher_order'] == ('post_fast' if arm == 'warm_post_sleep_P4' else 'pre_fast')
            boundary = (event_index + 1) * 4 if expected_events == 28 else 64
            due_slow = boundary % 16 == 0
            assert event['primary_boundary'] == boundary
            assert event['input_fast_count'] == 4
            assert event['input_slow_count'] == (16 if due_slow else boundary % 16)
            assert event['slow_level1_applied_before_teacher'] == due_slow
            assert event['fast_level0_applied_before_teacher'] == (arm == 'warm_post_sleep_P4')
            assert event['fast_level0_applied_prospectively'] == (arm != 'warm_post_sleep_P4')
            assert event['output_controller_counts'] == [0, 0 if due_slow else boundary % 16]
            seen_slots.append(event['receiver_slot'])
            assert event['receiver_slot'] == event_index + 1
            assert event['receiver_active_slots'] == list(range(event_index + 2))
            if not event['slow_level1_applied_before_teacher']:
                assert event['pending_slow_hash_before'] == event['pending_slow_hash_after']
        assert len(seen_slots) == len(set(seen_slots))
        if expected_events == 28:
            assert run['final_active_parameters']['active_slots_by_bank'] == [0, 29]
    assert sum(x['primary_updates'] for x in runs) == 2016
    assert sum(x['extra_replay_updates'] for x in runs) == 1008
    assert sum(x['sleep_Adam_updates'] for x in runs) == 4104
    for seed in SEEDS:
        pre = next(x for x in runs if x['seed'] == seed and x['arm'] == 'warm_pre_sleep_P4')
        post = next(x for x in runs if x['seed'] == seed and x['arm'] == 'warm_post_sleep_P4')
        assert pre['events'][0]['initial_student_fingerprint'] == post['events'][0]['initial_student_fingerprint']
    assert len(analysis['checkpoint_rows']) == 90
    assert [x['contrast'] for x in analysis['predefined_paired_contrasts']] == lock['paired_contrasts']
    for flags in (analysis['interpretation_flags'],):
        assert flags['development_only'] and flags['validation_adaptive_warm_budget']
        assert flags['confirmation_scored'] is False and flags['no_independent_confirmation_claim']

    changed = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only', '--',
        'src', 'results/agnews-qualified-v3', 'results/agnews-sleep-forensics-v3'])
    forbidden = ('.pt', '.bin', '.safetensors', '.parquet', '.arrow', '.npz', '.pkl', '.npy')
    assert not any(p.endswith(forbidden) for p in changed)
    assert not any('prompts-sealed' in p or 'labels-sealed' in p or 'private' in p for p in changed)

    def no_raw(value):
        if isinstance(value, dict):
            assert not ({'tokens', 'input_ids', 'raw_reference', 'sampled_candidate', 'teacher_data'} & set(value))
            for x in value.values():
                no_raw(x)
        elif isinstance(value, list):
            for x in value:
                no_raw(x)
    for path in changed:
        if path.startswith(BASE) and path.endswith('.json'):
            no_raw(read(path))
    print('PASS: staged v4 sources, metadata hash chains, all3 transfer gates, 18 runs/90 states, cost/isolation checks; older core/scientific evidence unchanged')


if __name__ == '__main__':
    verify()
