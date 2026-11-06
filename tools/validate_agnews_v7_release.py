"""Recompute the public v7 aggregate release without model assets or Torch."""
from pathlib import Path
import argparse,hashlib,json,re,subprocess,sys,tempfile
import xml.etree.ElementTree as ET

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());a=p.parse_args()
    root=a.root.resolve();base=Path('results/agnews-single-receiver-v7')
    assert root.is_dir()
    with tempfile.TemporaryDirectory(prefix='agnews-v7-reanalysis-') as tmp:
        td=Path(tmp)
        args=['--main-dir',str(base/'main-r2'),'--config','configs/agnews-single-receiver-v7-r2.json',
              '--qualification',str(base/'resume-qualification-r2/qualification.json'),
              '--cache-qualification',str(base/'cache-qualification-r3/qualification.json'),
              '--loss-path-qualification',str(base/'loss-path-qualification-r1/qualification.json'),
              '--qualification-work',str(base/'resume-qualification-r2/execution-work.json'),
              '--prior-qualification-failure',str(base/'resume-qualification-r1/qualification-failure.json'),
              '--prior-failure-work',str(base/'resume-qualification-r1/execution-work.json'),
              '--recovery-qualification',str(base/'main-r2/recovery-qualification-r1/qualification.json'),
              '--recovery-qualification-work',str(base/'main-r2/recovery-qualification-r1/execution-work.json'),
              '--source-root','.','--out-dir',str(td/'main')]
        subprocess.run([sys.executable,'tools/summarize_agnews_single_receiver_v7.py',*args],cwd=root,check=True)
        subprocess.run([sys.executable,'tools/summarize_agnews_v7_probes.py','--stress-dir',str(base/'stress-main-r1'),
            '--diagnostics-dir',str(base/'diagnostics-r1'),'--out-dir',str(td/'probe')],cwd=root,check=True)
        compared=[]
        for folder,target in (('main','analysis'),('probe','probe-analysis')):
            for f in sorted((td/folder).iterdir()):
                published=root/base/target/f.name
                if f.suffix=='.json':assert read(f)==read(published),published
                else:assert f.read_text(encoding='utf-8')==published.read_text(encoding='utf-8'),published
                compared.append(published.relative_to(root).as_posix())
    analysis=read(root/base/'analysis/analysis.json');stress=read(root/base/'stress-main-r1/result.json')
    diag=read(root/base/'diagnostics-r1/result.json');cp=read(root/base/'checkpoint-readback.json')
    assert analysis['accepted_complete_run'] and max(g['joint'] for g in analysis['joint_gate_counts'].values())<3
    assert stress['complete'] and stress['passed'] and stress['actual_CMS_ticks']==1536 and len(stress['rows'])==12
    assert diag['complete'] and diag['optimizer_updates']==0 and diag['independent_gradient_states']==93
    assert len(diag['gradient_rows'])==120 and len(diag['lesion_rows'])==120
    assert cp['passed'] and cp['checkpoint_count']==155 and cp['optimizer_updates']==0
    assert not any(x['confirmation_scored'] for x in (analysis,stress,diag))
    budget=read(root/base/'cpu-budget.json');assert sum(budget[k] for k in ('optimizer_attempts','functional_cms_attempts','controller_attempts'))==253
    for f in (root/base).rglob('*'):
        if f.is_file():
            assert f.suffix in ('.py','.json','.md','.png','.svg'),f
            assert f.stat().st_size<3_000_000,f
            if f.suffix=='.svg':
                text=f.read_text(encoding='utf-8');assert '<!DOCTYPE' not in text and '<script' not in text.lower()
                for e in ET.fromstring(text).iter():
                    for k,v in e.attrib.items():
                        if k.split('}')[-1]=='href':assert v.startswith('#'),(f,v)
    report=root/'reports/agnews-single-receiver-v7.md';text=report.read_text(encoding='utf-8');assert 'DIAGNOSTICS_PENDING' not in text
    for link in re.findall(r'\]\(([^)]+)\)',text):
        if not link.startswith(('https://','http://','#')):assert (report.parent/link.split('#')[0]).resolve().is_file(),link
    print(json.dumps({'passed':True,'aggregate_files_content_reproduced':len(compared),'runs':30,'stress_branches':12,
        'independent_gradient_states':93,'lesions':120,'checkpoint_readback':155,'confirmation_scored':False},sort_keys=True))

if __name__=='__main__':main()
