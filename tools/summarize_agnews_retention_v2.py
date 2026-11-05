"""Offline reanalysis of aggregate AG News v2 results; no model/data dependency."""
import argparse,csv,json,statistics
from pathlib import Path

ARMS=('frozen','sft','cms','cms_sleep','cms_extra_replay')
LABELS={'frozen':'Frozen','sft':'Per-step adapter SFT','cms':'CMS','cms_sleep':'CMS+Sleep (mix 0.5)',
        'cms_extra_replay':'CMS+extra labeled replay'}
EM='full_vocab_two_token_greedy_exact_match'


def avg_sd(values):return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'n_seeds':len(values)}
def cost(work):
    f=sum(v['forward_input_tokens']for v in work.values())
    g=sum(v['gradient_forward_input_tokens']for v in work.values())
    return {'forward_calls':sum(v['forward_calls']for v in work.values()),
            'padded_forward_input_tokens':f,'padded_gradient_forward_input_tokens':g,'fw_plus_2gradfw_proxy':f+2*g}


def summarize(evaluation,seal):
    if evaluation['schema']!='agnews-retention-postseal-evaluation-v2' or seal['complete']is not True:
        raise ValueError('completed v2 aggregate required')
    runs={f'{r["seed"]}/{r["arm"]}':r for r in seal['runs']}
    if set(evaluation['results'])!=set(runs) or len(runs)!=15:raise ValueError('all 15 paired runs required')
    individual,arms=[],{}
    for seed in (17,29,43):
        for arm in ARMS:
            key=f'{seed}/{arm}';trajectory=evaluation['results'][key];r=runs[key]
            before,after=trajectory[0]['metrics'],trajectory[-1]['metrics']
            if trajectory[0]['base_step']!=0 or trajectory[-1]['base_step']!=256:raise ValueError('bad endpoints')
            individual.append({'seed':seed,'arm':arm,
                **{f'{phase}_em':after[phase][EM]for phase in ('old','new','all')},
                'foundation_old_em':before['old'][EM],
                'old_forgetting':before['old'][EM]-after['old'][EM],
                'old_retention_ratio':None if before['old'][EM]==0 else after['old'][EM]/before['old'][EM],
                'answer_nll_nats_per_token':after['all']['answer_nll_nats_per_token'],
                'foundation_qualified':evaluation['foundation_qualifications'][str(seed)]['qualified_fixed_dev_criterion'],
                'new_primary_updates':r['primary_updates'],'extra_replay_updates':r['extra_replay_updates'],
                'sleep_adam_updates':r['sleep_adam_updates'],'new_phase_seconds':r['seconds'],**cost(r['work'])})
    for arm in ARMS:
        rows=[r for r in individual if r['arm']==arm]
        arms[arm]={'label':LABELS[arm],**{k:avg_sd([r[k]for r in rows])for k in (
            'old_em','new_em','all_em','foundation_old_em','old_forgetting','answer_nll_nats_per_token',
            'new_phase_seconds','fw_plus_2gradfw_proxy')},
            'new_primary_updates_total':sum(r['new_primary_updates']for r in rows),
            'extra_replay_updates_total':sum(r['extra_replay_updates']for r in rows),
            'sleep_adam_updates_total':sum(r['sleep_adam_updates']for r in rows)}
    ratios=[]
    for seed in (17,29,43):
        sleep=runs[f'{seed}/cms_sleep'];replay=runs[f'{seed}/cms_extra_replay']
        sw={k:v for k,v in sleep['work'].items() if '_event' in k}
        rw={k:v for k,v in replay['work'].items() if '_event' in k}
        ratios.append({'seed':seed,'sleep_extra_work':cost(sw),'replay_extra_work':cost(rw),
            'replay_to_sleep_additional_proxy_ratio':cost(rw)['fw_plus_2gradfw_proxy']/cost(sw)['fw_plus_2gradfw_proxy']})
    return {'schema':'agnews-retention-v2-offline-analysis','individual':individual,'arms':arms,
        'additional_cost_comparisons':ratios,'all_seeds_included':True,
        'n_unique_test_articles':512,'seeds_do_not_create_new_test_articles':True,
        'main_sgd_total':sum(r['updates']for r in seal['foundation_checkpoints'])+
            sum(r['primary_updates']+r['extra_replay_updates']for r in runs.values()),
        'main_sleep_adam_total':sum(r['sleep_adam_updates']for r in runs.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--seal',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--plots',action='store_true')
    args=p.parse_args();e=json.loads(args.evaluation.read_text(encoding='utf-8'));s=json.loads(args.seal.read_text(encoding='utf-8'))
    a=summarize(e,s);args.out_dir.mkdir(exist_ok=False)
    (args.out_dir/'analysis.json').write_text(json.dumps(a,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (args.out_dir/'per-seed-endpoints.csv').open('w',newline='',encoding='utf-8')as f:
        writer=csv.DictWriter(f,fieldnames=list(a['individual'][0]));writer.writeheader();writer.writerows(a['individual'])
    lines=['| Arm | Old EM % | New EM % | All EM % | Old forgetting pp |',
           '|---|---:|---:|---:|---:|']
    for arm,row in a['arms'].items():
        values=[f'{100*row[k]["mean"]:.2f} ± {100*row[k]["sample_sd"]:.2f}'for k in ('old_em','new_em','all_em','old_forgetting')]
        lines.append('| '+row['label']+' | '+' | '.join(values)+' |')
    (args.out_dir/'table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10.8,4),sharey=True,layout='constrained')
        for arm in ARMS:
            steps=[0,256]if arm=='frozen'else[0,64,128,192,256]
            for ax,phase in zip(axes,('old','new')):
                means,sds=[],[]
                for step in steps:
                    vals=[next(r['metrics'][phase][EM]*100 for r in e['results'][f'{seed}/{arm}']if r['base_step']==step)for seed in(17,29,43)]
                    means.append(statistics.mean(vals));sds.append(statistics.stdev(vals))
                ax.errorbar(steps,means,yerr=sds,marker='o',markersize=3,capsize=2,label=LABELS[arm],linewidth=1.4)
                ax.set(title=f'{phase.capitalize()} classes',xlabel='Primary new-task minibatches',xticks=[0,64,128,192,256],ylim=(-12,102))
                ax.grid(alpha=.2)
        axes[0].set_ylabel('Full-vocabulary two-token exact match (%)')
        axes[1].legend(loc='best',fontsize=7)
        qualified=sum(v['qualified_fixed_dev_criterion']for v in e['foundation_qualifications'].values())
        fig.suptitle('Llama 3.2-3B adapter study; three seeds, mean ± sample SD\n'
                     f'Foundation acquisition gate: {qualified}/3 passed; exploratory scores',fontsize=10)
        fig.savefig(args.out_dir/'retention-trajectory.png',dpi=180)
        fig.savefig(args.out_dir/'retention-trajectory.svg')
        plt.close(fig)
    print(json.dumps({'sgd':a['main_sgd_total'],'sleep_adam':a['main_sleep_adam_total'],
                      'foundation_qualified':{k:v['qualified_fixed_dev_criterion']for k,v in e['foundation_qualifications'].items()}}))


if __name__=='__main__':main()
