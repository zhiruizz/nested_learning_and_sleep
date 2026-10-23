"""BUNDLED_RESULTS_REANALYSIS: summaries from all bundled historical rows."""
import argparse
import csv
from pathlib import Path
from statistics import median

def summarize(root):
    def read(name):
        with (root/name).open(encoding='utf-8',newline='') as f:
            return list(csv.DictReader(f))
    acquisition=read('phase14_acquisition.csv')
    final=[r for r in acquisition if r['step']=='1024' and r['role']=='train']
    assert len(final)==6 and len(acquisition)==36
    assert len({r['history'] for r in final})==6
    complete=sum(r['fixed_final_gate']=='ESTABLISHED' for r in final)
    text=['# Historical table reanalysis', '', 'BUNDLED_RESULTS_REANALYSIS. No training or model forward.', '',
          f'Phase14 full acquisition gate: {complete}/{len(final)} histories. All {len(acquisition)} fixed scoring rows retained.', '',
          '| History | Train correct/N | Full gate |', '|---|---:|---|']
    text += [f"| {r['history']} | {r['correct']}/{r['count']} | {r['fixed_final_gate']} |" for r in final]
    bench=read('phase13_benchmark.csv')
    assert len(bench)==12
    ref={r['repeat']:float(r['full_path_seconds']) for r in bench if r['executor']=='REF'}
    ratios=[float(r['full_path_seconds'])/ref[r['repeat']] for r in bench if r['executor']=='CACHE' and r['status']=='completed']
    text += ['',f'CACHE: {len(ratios)}/3 complete pairs; median CACHE/REF={median(ratios):.9f}; reciprocal={1/median(ratios):.6f}x.',
             'This is an engineering repeat on a small repeated bank, not a general training speedup.', '',
             '| Executor | Completed/registered repeats | Missing or interrupted |', '|---|---:|---|']
    for mode in ('REF','CACHE','HOST','BOTH'):
        rows=[r for r in bench if r['executor']==mode]
        text.append(f"| {mode} | {sum(r['status']=='completed' for r in rows)}/{len(rows)} | "+'; '.join(r['status'] for r in rows if r['status']!='completed')+' |')
    text += ['', 'HOST did not meet the historical performance adoption rule. BOTH has incomplete performance pairing.',
             'Initial native pilot: all five group means were 0.25; four trajectories per group. This did not establish adequate acquisition.',
             'See results/README.md for denominators, versions, repeated development use and unavailable assets.', '']
    return '\n'.join(text)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('results'))
    p.add_argument('--output',type=Path)
    a=p.parse_args();value=summarize(a.results)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(value,encoding='utf-8')
    else:
        print(value)
if __name__=='__main__':
    main()
