"""A response-space metric for residual transfer; zero network/area queries."""
from pathlib import Path
import numpy as np
from secant_support import HERE,read,sha,save,array_save

SOURCE=HERE/'confirmation_s42_01'
OUT=HERE/'response_geometry_s42_01'
ARMS=('raw_scalar','input_kernel','scalar_response','mscs_response')


def solve(gram,cross,residual):
    e,v=np.linalg.eigh(gram)
    cutoff=1e-6*np.abs(e).max(axis=-1,keepdims=True)
    inv=np.zeros_like(e);mask=np.abs(e)>cutoff;inv[mask]=1/e[mask]
    inverse=np.einsum('nhi,ni,nki->nhk',v,inv,v)
    w=np.einsum('nch,nhk->nck',cross,inverse)
    correction=np.einsum('nch,hns->cns',w,residual)
    return correction,w


def response_kernel(candidate,history):
    # candidate [candidate,slice,response], history [history,slice,response].
    return (np.einsum('hnv,knv->nhk',history,history)**2,
            np.einsum('cnv,hnv->nch',candidate,history)**2)


def predictions(z,width):
    lo=1/width;hi=min(100.,1/(.01*width+1e-8))
    excess=np.maximum(z['root_cs']-lo,1e-10).astype(float)
    sd=z['stage_slope'].astype(float)/excess[None]
    sh=z['scalar__history_slope'].astype(float)
    q=z['root_m'][...,1:].astype(float);q/=q.sum(axis=-1,keepdims=True)
    md=(np.sqrt(q)[None]*z['logodds_slope']).reshape(*sd.shape[:2],-1)
    mh=(np.sqrt(q)[None]*z['mscs__history_slope']).reshape(*sh.shape[:2],-1)
    residual=z['scalar__residual'].astype(float)
    grids={'input_kernel':(z['scalar__gram'],z['scalar__cross']),
           'scalar_response':response_kernel(sd,sh),'mscs_response':response_kernel(md,mh)}
    all_scores={};arrays={'scalar_candidate':sd,'scalar_history':sh,'mscs_candidate':md,'mscs_history':mh,
                           'residual':residual,'root_cs':z['root_cs']}
    for arm in ARMS:
        correction=np.zeros_like(sd) if arm=='raw_scalar' else solve(*grids[arm],residual)[0]
        stages=np.clip(lo+excess[None]*np.exp(np.clip(sd+correction,-80,80)),lo,hi)
        all_scores[arm]=stages.mean((1,2)).tolist()
        arrays[arm+'__correction']=correction
        if arm!='raw_scalar':arrays[arm+'__gram'],arrays[arm+'__cross']=grids[arm]
    return all_scores,arrays


def main():
    from secant_support import compute_w_min,config_key
    OUT.mkdir(exist_ok=False)
    assert read(SOURCE/'validation.json')['passed']
    sources={str(p):sha(p) for p in (HERE/'response_geometry_protocol.txt',Path(__file__))}
    for p in SOURCE.rglob('*'):
        if p.is_file() and p.suffix in ('.json','.npz'):sources[str(p)]=sha(p)
    contexts=read(SOURCE/'plan.json')['contexts']
    save(OUT/'plan.json',dict(source_sha256=sources,contexts=contexts,arms=ARMS,
        phase='before replay prediction or metric evaluation',new_queries=0))
    records=[]
    for ctx in contexts:
        for path in sorted((SOURCE/'walks'/ctx['id']/'hybrid_secant').glob('decision_*.json')):
            d=read(path);step=d['step']
            with np.load(SOURCE/d['prediction_file']) as z:scores,arrays=predictions(z,compute_w_min(d['root']))
            prefix=Path('predictions')/ctx['id']/f'step_{step:02}'
            array_save(OUT/prefix.with_suffix('.npz'),**arrays)
            rec=dict(context=ctx['id'],bench=ctx['bench'],step=step,source_decision=str(path),
                chosen={a:np.argsort(s,kind='stable')[:2].tolist() for a,s in scores.items()},scores=scores,
                history_ids=d['history_ids'],source_label_indices=d['chosen'],
                array_file=str(prefix.with_suffix('.npz')),array_sha256=sha(OUT/prefix.with_suffix('.npz')))
            save(OUT/prefix.with_suffix('.json'),rec);records.append(rec)
    save(OUT/'frozen_predictions.json',dict(records=records,phase='ALL predictions before ANY current-outcome metric'))
    # Evaluation begins only after every score and top-two choice is frozen.
    rows=[]
    for rec in records:
        d=read(rec['source_decision']);r=read(Path(rec['source_decision']).with_name(f"round_{rec['step']:02}.json"))
        cp=d['root_cpi'];ids=d['chosen'];actual=[cp]+[t['cpi'] for t in r['trials']]
        for arm in ARMS:
            pred=[cp]+[rec['scores'][arm][i] for i in ids];loss=0;comparisons=0
            for i in range(3):
                for j in range(i+1,3):
                    if abs(actual[i]-actual[j])<=1e-6*cp:continue
                    comparisons+=1
                    loss+=1 if abs(pred[i]-pred[j])<=1e-6*cp else 0 if np.sign(pred[i]-pred[j])==np.sign(actual[i]-actual[j]) else 2
            errors=[abs(pred[i]-actual[i])/cp for i in (1,2)]
            rows.append(dict(context=rec['context'],bench=rec['bench'],step=rec['step'],arm=arm,
                loss2=loss,comparisons=comparisons,normalized_absolute_errors=errors,
                coverage=len(set(rec['chosen'][arm])&set(ids)),queries=2))
    groups={}
    for bench in ['all']+[ctx['bench'] for ctx in contexts]:
        groups[bench]={}
        for arm in ARMS:
            subset=[r for r in rows if r['arm']==arm and (bench=='all' or r['bench']==bench)]
            errors=sum([r['normalized_absolute_errors'] for r in subset],[])
            groups[bench][arm]=dict(loss2=sum(r['loss2'] for r in subset),comparisons=sum(r['comparisons'] for r in subset),
                mae=float(np.mean(errors)),max_error=float(max(errors)),coverage=sum(r['coverage'] for r in subset),labels=len(errors))
    g=groups['all'];m=g['mscs_response'];controls=('input_kernel','scalar_response')
    wins=[b for b,gg in groups.items() if b!='all' and gg['mscs_response']['loss2']<min(gg[a]['loss2'] for a in controls)]
    passed=all(m['loss2']<g[a]['loss2'] and m['mae']<g[a]['mae'] for a in controls) and len(wins)>=2 and any(b!='602.gcc_s' for b in wins)
    assert all(sha(p)==h for p,h in sources.items())
    save(OUT/'summary.json',dict(groups=groups,rows=rows,gate_passed=passed,rank_win_benchmarks=wins,new_queries=0,
        scope='development replay on hybrid-selected observations; no search benefit inference'))
    print(__import__('json').dumps(dict(groups=groups,gate_passed=passed),indent=2),flush=True)


if __name__=='__main__':main()
