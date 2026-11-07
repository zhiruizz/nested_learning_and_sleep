"""Plot aggregate-only CLINC development readouts; never load a model or dataset."""
import argparse
import json
from pathlib import Path

CONTEXTS=('FROZEN_BASE','TASK0_FOUNDATION','HOPE_TASK1','SLEEP_TASK1')
METHODS=('v8_greedy_generation','teacher_forced_sum','teacher_forced_length_mean')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    doc=json.loads(a.input.read_text(encoding='utf-8'))
    assert doc['schema']=='clinc-label-readout-diagnostic-v9' and doc['development_only'] is True
    assert doc['work']['optimizer_updates']==0 and doc['fixed_cohort']['rows']==60
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['svg.hashsalt']='clinc-v9-readout-v1'
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2),sharey=True)
    colors=('#245f8d','#d56028','#248669')
    labels=('Open generation','150-label sequence sum','150-label token mean')
    x=np.arange(4)
    width=.24
    for ax,group,title in zip(axes,('task0','task1'),('Task-0 classes (30 dev rows)','Task-1 classes (30 dev rows)')):
        for j,(method,color,label) in enumerate(zip(METHODS,colors,labels)):
            vals=[]
            for name in CONTEXTS:
                row=doc['metrics'][name][method][group]
                assert row['count']==30 and abs(row['accuracy']-row['correct']/30)<1e-12
                vals.append(100*row['correct']/30)
            bars=ax.bar(x+(j-1)*width,vals,width,label=label,color=color)
            ax.bar_label(bars,labels=[f'{value:.0f}' for value in vals],padding=2,fontsize=8)
        ax.set_xticks(x,['Frozen\nbase','Task-0\nfoundation','Task-1\nHOPE','Task-1\nfixed-mix Sleep'])
        ax.set_title(title,fontsize=12)
        ax.set_ylim(0,109)
        ax.set_yticks(range(0,101,20))
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Accuracy (%)')
    handles,legend=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='upper center',ncol=3,bbox_to_anchor=(.5,.935),frameon=False)
    fig.suptitle('CLINC V9: fixed-checkpoint readout diagnosis',fontsize=15,y=.99)
    fig.text(.5,.018,'One class-order seed; validation only. Candidate ranking adds the label dictionary and inference work; no parameter updates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.86))
    a.output_dir.mkdir(parents=True)
    stem=a.output_dir/'clinc-v9-readout-comparison'
    fig.savefig(stem.with_suffix('.png'),dpi=170,metadata={'Software':'NL/Sleep readout diagnosis'})
    fig.savefig(stem.with_suffix('.svg'),metadata={'Date':None})
    plt.close(fig)
    print(json.dumps({'figures':[stem.with_suffix('.png').name,stem.with_suffix('.svg').name]}))

if __name__=='__main__':main()
