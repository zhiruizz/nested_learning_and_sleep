"""Render development results from aggregate JSON only."""
from pathlib import Path
import argparse,json,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--stress',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
analysis=json.loads(a.analysis.read_text(encoding='utf-8'));stress=json.loads(a.stress.read_text(encoding='utf-8'))
arms=analysis['arms'];x=np.arange(len(arms));colors=['#326BA8','#D9842B']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(12,5.8))
for k,task in enumerate(('old','new')):
    values=[[analysis['endpoints'][arm]['per_seed'][str(s)]['validation'][task]['em']*100 for s in (17,29,43)] for arm in arms]
    ax.bar(x+(k-.5)*.35,[st.mean(v) for v in values],.35,label=task,color=colors[k],alpha=.8,
           yerr=[st.stdev(v) for v in values],capsize=3,error_kw={'linewidth':1})
    for i,v in enumerate(values):ax.scatter(np.repeat(x[i]+(k-.5)*.35,3)+np.array([-.045,0,.045]),v,s=12,color='#202020',zorder=3)
ax.set_xticks(x,[s.replace('_','\n') for s in arms]);ax.set_ylim(0,103);ax.set_ylabel('Validation exact match (%)')
ax.set_title('One-context offline distillation: all fixed seeds, step 1024\nDots: seeds 17 / 29 / 43; bars: mean ± sample SD; development only')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.18);fig.tight_layout()
for ext in ('png','svg'):fig.savefig(a.out/f'endpoints.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.4,4.8),sharey=True)
for ax,arm in zip(axes,('split_receiver','split_shared')):
    for k,task in enumerate(('old','new')):
        for writes,style in ((True,'-'),(False,'--')):
            rows=[r for r in stress['rows'] if r['source_arm']==arm and r['base_writes']==writes]
            steps=[r['step'] for r in rows[0]['evaluations']]
            vals=[[r['evaluations'][j]['validation'][task]['em']*100 for r in rows] for j in range(len(steps))]
            ax.plot(steps,[st.mean(v) for v in vals],style,color=colors[k],marker='o',markersize=3,
                    label=f'{task}, base {"apply" if writes else "skip"}')
    ax.set_title(arm);ax.set_xlabel('New-only CMS ticks');ax.set_xticks([0,16,64,128]);ax.set_ylim(0,100);ax.grid(alpha=.18)
axes[0].set_ylabel('Validation exact match (%)');axes[1].legend(frameon=False,fontsize=9,loc='lower left')
fig.suptitle('Frozen slow experts do not prevent subsequent interference\nMean of three fixed seeds; development diagnostic, not a third task');fig.tight_layout()
for ext in ('png','svg'):fig.savefig(a.out/f'wake-stress.{ext}',dpi=180)
plt.close(fig)
import xml.etree.ElementTree as ET
ET.register_namespace('','http://www.w3.org/2000/svg')
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
def signature(e):
    return (e.tag,tuple(sorted(e.attrib.items())),(e.text or '').strip(),(e.tail or '').strip(),tuple(signature(c) for c in e))
for path in a.out.glob('*.svg'):
    old=path.read_text(encoding='utf-8');tree=ET.fromstring(old);ET.indent(tree,space=' ')
    new=ET.tostring(tree,encoding='unicode')+'\n'
    assert signature(ET.fromstring(old))==signature(ET.fromstring(new))
    path.write_text(new,encoding='utf-8',newline='\n')
print(a.out)
