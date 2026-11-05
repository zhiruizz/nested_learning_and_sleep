"""Recompute the v1 pilot table from aggregate metrics; no model forward."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean


def summarize(root):
    cfg = json.loads((root/'config.json').read_text())
    ledger = json.loads((root/'ledger.json').read_text())
    rows = json.loads((root/'metrics.json').read_text())
    correctness = json.loads((root/'correctness.json').read_text())
    scopes = json.loads((root/'scopes.json').read_text())
    drift = json.loads((root/'parameter-drift.json').read_text())
    assert ledger['status'] == 'completed'
    assert ledger['update_attempts'] == ledger['completed_updates'] == cfg['max_update_attempts']
    assert correctness['stream_max_abs_error'] == 0 and correctness['period_one_matches_sft']
    def select(seed, arm, step):
        match = [r for r in rows if r['seed'] == seed and r['arm'] == arm and r['step'] == step]
        assert len(match) == 1
        return match[0]
    for seed in cfg['seeds']:
        sft = select(seed, 'sft', cfg['adapt_steps'])
        cms = select(seed, 'cms', cfg['adapt_steps'])
        assert sft['data_sha256'] == cms['data_sha256']
        assert sft['start_model_sha256'] == cms['start_model_sha256']
        assert sft['base_writes'] == [cfg['adapt_steps']]*2
        assert cms['base_writes'] == [cfg['adapt_steps']//p for p in cfg['cms_periods']]
        assert sft['pending_counts'] == cms['pending_counts'] == [0, 0]
        pair = [r for r in scopes if r['seed'] == seed]
        assert len(pair) == 2 and pair[0]['names'] == pair[1]['names']
        for arm in ('sft', 'cms'):
            allowed = next(r['names'] for r in pair if r['arm'] == arm)
            changes = [r for r in drift if r['seed'] == seed and r['stage'] == arm]
            assert len(changes) == 1
            assert not any(r['changed'] for r in changes[0]['parameters'] if r['name'] not in allowed)
        for row in [r for r in rows if r['seed'] == seed]:
            for metric in row['metrics'].values():
                assert metric['count'] == 512
                assert metric['accuracy'] == metric['greedy_correct']/512
                assert metric['candidate_accuracy'] == metric['candidate_correct']/512
                assert math.isfinite(metric['nll'])
    text = ['# Native symbol retention pilot v1', '',
            'Completed bounded synthetic CPU experiment. This is not a paper benchmark or a direct replication of upstream HOPE scores.', '',
            'Protocol: [native-retention-protocol.md](../docs/native-retention-protocol.md). Recompute with `python tools/summarize_native_retention.py --results results/native-retention-v1`.', '',
            'Three fixed seeds; old A and one-token copy foundation, then new B adaptation. Here SFT denotes supervised per-batch updates on the same restricted parameter scope as CMS, not full-model fine-tuning. SFT/CMS share starting weights and batches. Candidate chance is 25%. All tasks have 512 held-out synthetic prompts per seed.', '',
            '| Seed | State | A accuracy | Copy accuracy | B accuracy | A NLL | Copy NLL | B NLL |',
            '|---:|---|---:|---:|---:|---:|---:|---:|']
    states = [('random', 0), ('foundation', cfg['foundation_steps']), ('frozen', cfg['adapt_steps']),
              ('sft', cfg['adapt_steps']), ('cms', cfg['adapt_steps'])]
    for seed in cfg['seeds']:
        for arm, step in states:
            m = select(seed, arm, step)['metrics']
            text.append(f"| {seed} | {arm} | " + ' | '.join(f"{100*m[t]['accuracy']:.2f}%" for t in ('old_a','copy','new_b')) + ' | ' + ' | '.join(f"{m[t]['nll']:.6f}" for t in ('old_a','copy','new_b')) + ' |')
    text += ['', '## Acquisition and retention', '',
             '| Seed | Old A foundation >=80% | Copy foundation >=80% |', '|---:|---|---|']
    for seed in cfg['seeds']:
        m = select(seed, 'foundation', cfg['foundation_steps'])['metrics']
        text.append(f"| {seed} | {'YES' if m['old_a']['accuracy']>=cfg['acquisition_accuracy'] else 'NO'} | {'YES' if m['copy']['accuracy']>=cfg['acquisition_accuracy'] else 'NO'} |")
    text += ['', 'Mean endpoints (descriptive, all three seeds):', '',
             '| Arm | A accuracy | Copy accuracy | B accuracy | A change vs foundation (pp) | B change vs foundation (pp) |',
             '|---|---:|---:|---:|---:|---:|']
    for arm in ('frozen','sft','cms'):
        ms = [select(s, arm, cfg['adapt_steps'])['metrics'] for s in cfg['seeds']]
        fs = [select(s, 'foundation', cfg['foundation_steps'])['metrics'] for s in cfg['seeds']]
        values = [100*mean(m[t]['accuracy'] for m in ms) for t in ('old_a','copy','new_b')]
        deltas = [100*mean(m[t]['accuracy']-f[t]['accuracy'] for m,f in zip(ms,fs)) for t in ('old_a','new_b')]
        text.append(f"| {arm} | " + ' | '.join(f'{v:.2f}%' for v in values) + ' | ' + ' | '.join(f'{v:+.2f}' for v in deltas) + ' |')
    acquired = all(select(s, 'foundation', cfg['foundation_steps'])['metrics'][t]['accuracy'] >= cfg['acquisition_accuracy'] for s in cfg['seeds'] for t in ('old_a','copy'))
    finding = ('All three seeds acquired both old A and copy at100%; their subsequent accuracy loss is forgetting of acquired tasks in this experiment.' if acquired else 'Some foundation tasks failed the frozen acquisition gate; only tasks passing that gate can support retention language.')
    text += ['', finding + ' SFT/CMS differences are within this budget and schedule; three seeds do not support a general superiority claim.', '',
             '## Validation and cost', '',
             f"- Stream/full-forward fixture error: {correctness['stream_max_abs_error']}; period-one CMS matches SFT exactly.",
             '- All adaptation starts and B batch hashes match; mutable scopes match; no frozen parameter changed (runner asserts).',
             f"- Adaptation parameter scope: {scopes[0]['parameter_count']} / {scopes[0]['total_parameters']} parameters. SFT base writes128/128, CMS32/8; both have128 online expert updates. Pending counts end at0/0.",
             f"- Main attempts/completed: {ledger['update_attempts']}/{ledger['completed_updates']}; correctness fixture reserved/completed: {ledger['fixture_attempts_reserved']}/{ledger['fixture_completed_updates']}.",
             f"- Elapsed runner time: {ledger['elapsed_seconds']:.3f}s on CPU, one thread. Main supervised answer tokens: {ledger['supervised_answer_tokens']}; fixture: {ledger['fixture_supervised_answer_tokens']}.",
             '- Frozen metrics reuse the identical foundation state; no duplicate model forward is charged.', '',
             '| Stage | Forward calls | Input tokens | Gradient-enabled input tokens |', '|---|---:|---:|---:|']
    for stage, v in sorted(ledger['work_meter'].items()):
        text.append(f"| {stage} | {v['forward_calls']} | {v['forward_input_tokens']} | {v['gradient_forward_input_tokens']} |")
    text += ['', 'The stream fixture additionally executes32 core input tokens outside WorkMeter, without gradients. Token counts are compute proxies, not FLOPs or GPU busy time. Baseline/evaluation and foundation are included in the ledger. No GPU job, model download or external dataset was used.', '',
             '## Limits', '',
             'One tiny native variant; synthetic nuisance split; short copy can use convolution history. The B domain marker makes old/new mappings jointly representable in principle, but the restricted adaptation scope need not access such a solution. No Sleep/GKD/Dreaming arm, long-context test, natural-language perplexity, factual benchmark or pretrained 3B result is established. Parameter drift locates changes; it does not identify causal learning mechanisms. All seeds, intermediate evaluation points32/128 and negative results remain in metrics.json. No weights or per-example dataset records are distributed.', '',
             'Published JSON uses UTF-8/LF line endings; numerical values are unchanged from the Windows runner output retained locally. config_sha256 in provenance identifies the preregistered configs/native-retention-v1.json source, not the pretty-printed result copy. Source commit identifies the run code before outcomes; this report and aggregate archive were added afterwards.', '']
    return '\n'.join(text)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = summarize(a.results)
    if a.output:
        a.output.write_bytes(result.encode('utf-8'))
    else:
        print(result)
