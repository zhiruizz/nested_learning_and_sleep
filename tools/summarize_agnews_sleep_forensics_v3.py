"""Offline analysis of exploratory first-boundary/route diagnostics, not a test."""
from pathlib import Path
import argparse,hashlib,json,statistics
EM='full_vocab_two_token_greedy_exact_match'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean_sd(v):return {'n':len(v),'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)if len(v)>1 else None}
def summarize(source,plan_path):
 d=read(source);plan=read(plan_path)
 assert d['schema']=='agnews-sleep-forensics-v3'and d['exploratory_not_confirmation']and d['heldout_read']is False
 assert d['source_sha256']==plan['source_sha256'] and plan['heldout_input']is False
 assert d['supervised_replay_steps']==192 and d['sleep_adam_replay_steps']==72
 assert {r['seed']for r in d['results']}=={17,29,43}and len(d['results'])==3
 trajectories={};routes={};sample_groups={};states=[]
 for r in d['results']:
  assert all(r['first_sleep_exact_replay_matches'].values())
  if r['seed']in(29,43):assert r['CMS64_control_exact_replay_matches_when_available']
  assert len(r['first_event_generated_target_diagnostics'])==24
  expected_stages={'foundation_step0','foundation_disable_fast_bank','wake_step8','wake_step16','wake_step32','wake_step63','boundary64_after_collect_before_base_commits','boundary64_CMS_no_reset_control','first_teacher_after_slow_before_fast_reset','first_student_after_fast_reset_mix_before_training','first_student_after_16_sleep_updates','first_student_after_24_sleep_updates','committed_first_sleep_step64'}
  name_of=lambda x:x['stage'].split(f'diagnostic_seed{r["seed"]}_',1)[1]
  assert {name_of(x)for x in r['validation_trajectory_and_first_boundary']}==expected_stages and len(r['validation_trajectory_and_first_boundary'])==13
  assert {name_of(x)for x in r['endpoint_route_probes_validation_only']}=={f'endpoint_{x}'for x in plan['final_route_probes']} and len(r['endpoint_route_probes_validation_only'])==5
  indexed={name_of(x):x for x in r['validation_trajectory_and_first_boundary']}
  assert indexed['first_student_after_24_sleep_updates']['metrics']==indexed['committed_first_sleep_step64']['metrics']
  for x in r['validation_trajectory_and_first_boundary']+r['endpoint_route_probes_validation_only']:
   assert x['observation_unchanged_weights_policy_rng']and x['independent_autoregressive_check']['exact_agreement']
   assert x['independent_autoregressive_check']['metered_model_forward_calls']==6
   assert all(x['metrics'][g]['examples']==n for g,n in {'old':256,'new':256,'all':512,**{f'class{i}':128 for i in range(4)}}.items())
   name=x['stage'].split(f'diagnostic_seed{r["seed"]}_',1)[1]
   target=routes if name.startswith('endpoint_')else trajectories
   target.setdefault(name,[]).append({'seed':r['seed'],'old_EM':x['metrics']['old'][EM],'new_EM':x['metrics']['new'][EM],
    'old_marker_accuracy':x['metrics']['old']['marker_accuracy'],'new_marker_accuracy':x['metrics']['new']['marker_accuracy'],
    'old_class_NLL':x['metrics']['old']['class_nll'],'new_class_NLL':x['metrics']['new']['class_nll'],
    'route_and_component_diagnostics':x['route_and_component_diagnostics']})
   states.append(x['independent_autoregressive_check'])
  for x in r['first_event_generated_target_diagnostics']:
   # Probabilities under a sampled incorrect marker are a different context;
   # retain the raw aggregate row, but stratify valid-marker targets separately.
   for group in ('all_markers','correct_marker_only'):
    if group=='correct_marker_only'and not x['sample_marker_correct']:continue
    sample_groups.setdefault(f'{x["phase"]}/{x["kind"]}/{group}',[]).append(x)
 def metrics(rows):return {k:mean_sd([r[k]for r in rows])for k in('old_EM','new_EM','old_marker_accuracy','new_marker_accuracy','old_class_NLL','new_class_NLL')}
 sample_summary={}
 for group,rows in sample_groups.items():
  sample_summary[group]={'samples':len(rows),'sample_marker_hits':sum(x['sample_marker_correct']for x in rows),
   'sampled_gold_class_hits':sum(x['sampled_class_correct']for x in rows),
   'teacher_greedy_gold_class_hits':sum(x['teacher_greedy_class_correct_given_actual_prefix']for x in rows),
   **{k:statistics.mean(x[k]for x in rows)for k in('teacher_gold_class_probability_given_actual_prefix','teacher_old_code_mass_given_actual_prefix','teacher_new_code_mass_given_actual_prefix','loss_before_update','same_sample_objective_after_update','teacher_student_KL_after_update')},
   'same_sample_objective_decreased':sum(x['same_sample_objective_after_update']<x['loss_before_update']for x in rows)}
 return {'schema':'agnews-sleep-forensics-analysis-v3','diagnostic_source_sha256':d['source_sha256'],'diagnostic_result_sha256':sha(source),'diagnostic_plan_sha256':sha(plan_path),
  'exploratory_not_confirmation':True,'heldout_read':False,
  'trajectory_by_stage':{k:{'per_seed':v,'mean_sample_sd':metrics(v)}for k,v in trajectories.items()},
  'endpoint_route_probes':{k:{'per_seed':v,'mean_sample_sd':metrics(v)}for k,v in routes.items()},
  'generated_teacher_target_diagnostics':sample_summary,
  'exact_first64_replay_all3':True,'independent_greedy_checks':{'states':len(states),'comparisons':sum(x['examples']for x in states),'all_exactly_match_frozen_scorer':True,'extra_model_forwards_are_metered':True},
  'all3_post24_student_metrics_exactly_equal_committed_metrics':True,
  'schedule_audit':{'fast_base_period':4,'slow_base_period':16,'Sleep_period':64,'fast_base_commits_before_first_Sleep':list(range(4,64,4)),
    'slow_base_commits_before_first_Sleep':list(range(16,64,16)),'Sleep_precedes_each_fast_base_commit':False,'teacher_additionally_contains_boundary64_slow_commit':True},
  'cost':{'supervised_updates':192,'sleep_Adam_updates':72,'elapsed_seconds':d['seconds'],'work':d['work'],
   'time_limit_enforcement':plan['time_enforcement'],'forward_token_proxy_omits_direct_low_rank_norm_hook_calls':True},
  'scope':'Post-v3 exploratory replay and fixed routing probes on existing train-derived validation; no new confirmation, method selection, or heldout scoring.'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--plots',action='store_true');a=p.parse_args()
 result=summarize(a.diagnostic,a.plan);a.out_dir.mkdir(exist_ok=False,parents=True)
 (a.out_dir/'analysis.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
 for kind,key in(('trajectory','trajectory_by_stage'),('routing','endpoint_route_probes')):
  rows=['| Stage / policy | Old EM % | New EM % |','|---|---:|---:|']
  for name,value in result[key].items():
   m=value['mean_sample_sd'];rows.append(f'| {name} | {100*m["old_EM"]["mean"]:.2f} ± {100*m["old_EM"]["sample_sd"]:.2f} | {100*m["new_EM"]["mean"]:.2f} ± {100*m["new_EM"]["sample_sd"]:.2f} |')
  (a.out_dir/(kind+'-table.md')).write_text('\n'.join(rows)+'\n',encoding='utf-8',newline='\n')
 if a.plots:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),sharey=True)
  trajectory=result['trajectory_by_stage'];left=['foundation_step0','wake_step8','wake_step16','wake_step32','wake_step63','first_teacher_after_slow_before_fast_reset']
  right=['first_teacher_after_slow_before_fast_reset','first_student_after_fast_reset_mix_before_training','first_student_after_16_sleep_updates','first_student_after_24_sleep_updates']
  for metric,color,label in [('old_EM','#0072B2','Old classes'),('new_EM','#D55E00','New classes')]:
   for ax,keys,xs in ((axes[0],left,[0,8,16,32,63,64]),(axes[1],right,list(range(4)))):
    means=[100*trajectory[x]['mean_sample_sd'][metric]['mean']for x in keys];sd=[100*trajectory[x]['mean_sample_sd'][metric]['sample_sd']for x in keys]
    ax.plot(xs,means,color=color,label=label,marker='o',linewidth=2)
    ax.fill_between(xs,[max(0,m-s)for m,s in zip(means,sd)],[min(100,m+s)for m,s in zip(means,sd)],color=color,alpha=.12)
  axes[0].set(title='New-task wake before first Sleep',xlabel='Primary new-task minibatches',xticks=[0,8,16,32,64],ylabel='Validation two-token EM (%)')
  axes[1].set(title='First consolidation transaction',xticks=range(4));axes[1].set_xticklabels(['Teacher','Student\ninitial','After GKD\n16 updates','After LTI\n24 total'])
  for ax in axes:ax.set_ylim(0,100);ax.grid(alpha=.2)
  handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
  fig.suptitle('Exploratory v3 validation replay · 3 seeds · mean ± sample SD',fontsize=12)
  fig.tight_layout(rect=(0,.09,1,.93));fig.savefig(a.out_dir/'first-boundary.png',dpi=180);fig.savefig(a.out_dir/'first-boundary.svg');plt.close(fig)
 print(json.dumps({'status':'EXPLORATORY_SUMMARY_COMPLETE','greedy_checks':result['independent_greedy_checks'],'cost_sgd':192,'cost_Adam':72}))
if __name__=='__main__':main()
