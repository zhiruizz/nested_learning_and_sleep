"""Plot audited development counts; no model/data/optimization operations."""
import argparse
import json
from pathlib import Path


def collect_development_plot_values(summary):
    if summary.get('schema')!='clinc-v8-two-task-development-summary-v1' or not summary.get('complete') or summary.get('test_payload_read') is not False:
        raise ValueError('expected complete, development-only reanalysis')
    rows=summary['rows']
    if [r['arm'] for r in rows]!=['HOPE','SLEEP_MIX','GOLD_CE_MIX']:
        raise ValueError('unexpected arm order')
    values={cohort:[r['metrics'][cohort]['accuracy']*100 for r in rows] for cohort in ('old','current')}
    foundation=summary['shared_foundation']['old_development']['accuracy']*100
    counts={cohort:[[r['metrics'][cohort]['correct'],r['metrics'][cohort]['count']] for r in rows]
            for cohort in ('old','current')}
    return {'values_percent':values,'counts':counts,'foundation_percent':foundation}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True);p.add_argument('--out-dir',required=True)
    a=p.parse_args();s=json.loads(Path(a.summary).read_text(encoding='utf-8'))
    plot_data=collect_development_plot_values(s)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.hashsalt':'clinc-v8-development-1','axes.spines.top':False,'axes.spines.right':False})
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    labels=['No Sleep','Sleep + fixed mix','Gold CE + fixed mix']
    colors=['#4C78A8','#E17C42','#56A18C']
    rows=s['rows']
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.8),sharey=True)
    for ax,cohort,title in zip(axes,('old','current'),('Old task after learning task 1','New task after learning task 1')):
        values=plot_data['values_percent'][cohort]
        ax.bar(range(3),values,color=colors,width=.62,zorder=3)
        for i,(r,v) in enumerate(zip(rows,values)):
            m=r['metrics'][cohort]
            ax.text(i,v+2.2,f"{v:.2f}%\n{m['correct']}/{m['count']}",ha='center',va='bottom',fontsize=10)
        ax.set_xticks(range(3),[x.replace(' + ','\n+ ') for x in labels],fontsize=10)
        ax.set_ylim(0,100);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2,zorder=0)
        ax.set_title(title,fontsize=12,pad=12)
    foundation=plot_data['foundation_percent']
    axes[0].axhline(foundation,color='#555555',lw=1.2,ls='--',zorder=2)
    axes[0].text(-.35,foundation+2,f'Shared old foundation: {foundation:.2f}%',fontsize=9,color='#444444')
    axes[0].set_ylabel('Clean development accuracy (%)')
    fig.suptitle('CLINC150: two-task development results',fontsize=15,y=.98)
    fig.text(.5,.902,'Seed 17; 300 examples per cohort; exploratory, not final test or paper replication',ha='center',fontsize=10,color='#555555')
    fig.text(.5,.025,'Gold CE uses additional labeled targets. Controlled mix adds expert computation. No error bars: one class-order seed.',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.78,bottom=.2,wspace=.17,left=.08,right=.98)
    for ext in ('svg','png'):
        path=out/f'clinc-v8-development-endpoints.{ext}'
        if path.exists():raise FileExistsError(path)
        fig.savefig(path,dpi=180,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    print(json.dumps({'figures':2,'out_dir':str(out)}))


if __name__=='__main__':main()
