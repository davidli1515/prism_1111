"""Export the post-hoc frozen-surrogate witness, with its limitations on the figure."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;SRC=HERE/'interaction_diamond_s42_01';OUT=HERE/'decoupling_figure_01'


def main():
    OUT.mkdir(exist_ok=False)
    s=json.loads((SRC/'summary.json').read_text());p=json.loads((SRC/'frozen_predictions.json').read_text())
    assert json.loads((SRC/'validation.json').read_text())['passed']
    actual=np.array([r['cpi'] for r in s['corners']]);root=actual[0]
    gain={k:100*(1-np.array(v)[1:]/root) for k,v in [('Scalar tangent',p['scalar_cpis']),('MSCS tangent',p['mscs_cpis']),('Full forward',actual)]}
    accounts=np.array([r['account'] for r in s['corners']]);changes=accounts[1:,1:]-accounts[0,1:]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13.2,3.6),gridspec_kw={'width_ratios':[1,1.6,1.6]})
    ax=axes[0];matrix=np.array([[actual[0],actual[1]],[actual[2],actual[3]]])
    ax.imshow(matrix,cmap='RdBu_r',vmin=3.43,vmax=3.51)
    for row in range(2):
        for col in range(2):ax.text(col,row,f'{matrix[row,col]:.4f}',ha='center',va='center',color='black',fontweight='bold')
    ax.set_xticks([0,1],['LQ 40','LQ 64']);ax.set_yticks([0,1],['IQ 64','IQ 96'])
    ax.set_title('Measured by full surrogate\nlower CPI is better')
    names=['LQ only','IQ only','LQ + IQ'];x=np.arange(3);colors=['#888888','#3479b5','#d5832a'];bw=.24
    ax=axes[1]
    for j,(name,values) in enumerate(gain.items()):ax.bar(x+(j-1)*bw,values,width=bw,label=name,color=colors[j])
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,names);ax.set_ylabel('CPI reduction from root (%)');ax.set_title('Predicting the sign of a finite move')
    ax.legend(frameon=False,fontsize=8,loc='lower left')
    ax=axes[2];positive=np.zeros(3);negative=np.zeros(3)
    for j,(name,color) in enumerate(zip(['FP','BP','SP'],['#d95f02','#1b9e77','#7570b3'])):
        vals=changes[:,j];base=np.where(vals>=0,positive,negative);ax.bar(x,vals,bottom=base,label=name,color=color,width=.58)
        positive+=np.maximum(vals,0);negative+=np.minimum(vals,0)
    ax.plot(x,changes.sum(1),'ko',label='Net change',ms=4);ax.axhline(0,color='black',lw=.7)
    ax.set_xticks(x,names);ax.set_ylabel('Change in CPI account');ax.set_title('Opposing component responses')
    ax.legend(frameon=False,fontsize=8,ncol=2)
    fig.suptitle('roms: two individually harmful increases form a beneficial joint move',fontsize=13,y=1.02)
    fig.text(.5,-.025,'Post-hoc diagnostic, frozen PRISM surrogate (not gem5). Stage accounts are averaged. No claim of universal search superiority.',ha='center',fontsize=9)
    fig.tight_layout()
    for suffix in ('pdf','svg','png'):fig.savefig(OUT/f'decoupling_witness.{suffix}',bbox_inches='tight',dpi=180)
    plt.close(fig)
    with (OUT/'data.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['action','scalar_predicted_cpi','mscs_predicted_cpi','full_forward_cpi','area','FP_change','BP_change','SP_change'])
        for i,name in enumerate(names,1):w.writerow([name,p['scalar_cpis'][i],p['mscs_cpis'][i],actual[i],p['areas'][i],*changes[i-1]])


if __name__=='__main__':main()
