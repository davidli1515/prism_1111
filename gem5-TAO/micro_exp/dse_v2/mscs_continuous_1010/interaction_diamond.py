"""One missing corner separates prediction failure, interaction and move-grid effects."""
import json,sqlite3,time
import numpy as np
import torch
from quadratic_support import HERE,ROOT,read,sha,save,array_save,Inputs,FeatureCache,LocalModel,load_model,exact,compute_w_min
from core import cpis
from width_priority_support import predict
from matched_support import label_id
import run_adaptive as area_runner

SRC=HERE/'priority_walk_s42_01';OUT=HERE/'interaction_diamond_s42_01';BENCH='654.roms_s'


def main():
    start=time.perf_counter();OUT.mkdir(exist_ok=False);assert read(SRC/'validation.json')['passed']
    oldfile=SRC/'walks'/BENCH/'mscs_raw/decision_04.json';d=read(oldfile);root=d['root']
    scalar4=read(SRC/'walks'/BENCH/'scalar_raw/round_03.json');scalar5=read(SRC/'walks'/BENCH/'scalar_raw/round_04.json')
    configs=[dict(root),dict(root,LQEntries=64),dict(root,numIQEntries=96),dict(root,LQEntries=64,numIQEntries=96)]
    assert configs[1]==scalar4['next_root'] and configs[3]==scalar5['next_root']
    sources=dict(read(SRC/'plan.json')['source_sha256'])
    for p in [*SRC.rglob('*.json'),*SRC.rglob('*.npz'),HERE/'interaction_diamond.py',HERE/'interaction_diamond_protocol.txt']:
        sources[str(p)]=sha(p)
    save(OUT/'plan.json',dict(bench=BENCH,configs=configs,source_sha256=sources,scope='post-hoc 2x2 contrast; not blind efficacy evidence'))
    cache=FeatureCache(Inputs(BENCH,64));model=load_model();local=LocalModel(model,cache.get(root),compute_w_min(root),'local')
    xs=torch.stack([cache.get(c) for c in configs]);delta=xs-local.x[None];de=torch.einsum('nskd,cnd->cnsk',local.gell,delta)
    pred=predict(local.ell,de,[compute_w_min(c) for c in configs])
    chord=delta[1]+delta[2];odds=torch.einsum('nskd,nd->nsk',local.gell,chord)
    additive=predict(local.ell,odds[None],[compute_w_min(root)])
    area_runner.OUT=OUT;area=area_runner.Areas()
    for p in ROOT.glob('mscs_*_1010/*/area_store/areas.sqlite'):
        if OUT not in p.parents:area.old.append(sqlite3.connect(f'file:{p}?mode=ro',uri=True))
    areas=[area.get(c) for c in configs]
    array_save(OUT/'predictions.npz',root_m=local.m,root_ell=local.ell,logodds_slope=de,predicted_m=pred['predicted_m'])
    frozen=dict(scalar_cpis=pred['scalar_cpi'].tolist(),mscs_cpis=pred['mscs_cpi'].tolist(),areas=areas,
        additive_joint_scalar_cpi=float(additive['scalar_cpi'][0]),additive_joint_mscs_cpi=float(additive['mscs_cpi'][0]),
        joint_input_chord_error=float(torch.linalg.norm(delta[3]-chord)),joint_input_delta_norm=float(torch.linalg.norm(delta[3])),
        in_original_candidate_pool=[any(c==a['config'] for a in d['candidates']) for c in configs],prediction_sha256=sha(OUT/'predictions.npz'))
    save(OUT/'frozen_predictions.json',frozen);labels=[];new=0;reused=0
    for i,c in enumerate(configs):
        lid=label_id(BENCH,c);old=SRC/'truth'/BENCH/f'{lid}.npz'
        if old.exists():
            with np.load(old) as z:mm=torch.from_numpy(z['m'].copy())
            cp=float(cpis(mm,compute_w_min(c)).mean());reused+=1;origin='existing_paid_corner'
        else:cp,mm=exact(model,xs[i],compute_w_min(c));new+=1;origin='new_missing_corner'
        array_save(OUT/f'corner_{i}.npz',m=mm)
        labels.append(dict(config=c,cpi=cp,area=areas[i],origin=origin,account=(cpis(mm,compute_w_min(c))[...,None]*mm).mean((0,1)).tolist()))
    cp=np.array([r['cpi'] for r in labels]);interaction=float(cp[3]-cp[1]-cp[2]+cp[0])
    summary=dict(corners=labels,labels=['00','10_LQ','01_IQ','11_joint'],root_cpi=float(cp[0]),
        interaction_cpi=interaction,normalized_interaction=interaction/cp[0],
        lq_effect_at_low_iq=float(cp[1]-cp[0]),lq_effect_at_high_iq=float(cp[3]-cp[2]),
        iq_effect_at_low_lq=float(cp[2]-cp[0]),iq_effect_at_high_lq=float(cp[3]-cp[1]),
        new_candidate_forwards=new,reused_corner_labels=reused,root_forwards=1,backward_outputs=12,
        area_counts=dict(area.counts),area_failures=area.store.n_fail,wall_seconds=time.perf_counter()-start)
    assert all(sha(p)==h for p,h in sources.items())
    # Independently check class accounts and the four-corner contrast.
    from scipy.special import softmax,logsumexp,expit
    with np.load(OUT/'predictions.npz') as z:
        ell=z['root_ell'].astype(float);slope=z['logodds_slope'].astype(float)
        u=logsumexp(ell,-1)[None]+(softmax(ell,-1)[None]*slope).sum(-1)
        sc=np.minimum(100,1/(np.maximum(expit(-u),.01)*compute_w_min(root)+1e-8)).mean((1,2))
        ll=ell[None]+slope;mm=softmax(np.concatenate([np.zeros_like(ll[...,:1]),ll],-1),-1)
        mc=np.minimum(100,1/(np.maximum(mm[...,0],.01)*compute_w_min(root)+1e-8)).mean((1,2))
        np.testing.assert_allclose(sc,frozen['scalar_cpis'],rtol=3e-6);np.testing.assert_allclose(mc,frozen['mscs_cpis'],rtol=3e-6)
    np.testing.assert_allclose([sum(r['account']) for r in labels],cp,rtol=3e-6)
    assert np.isclose(summary['iq_effect_at_high_lq']-summary['iq_effect_at_low_lq'],interaction)
    assert np.isclose(summary['lq_effect_at_high_iq']-summary['lq_effect_at_low_iq'],interaction)
    save(OUT/'summary.json',summary);save(OUT/'validation.json',dict(passed=True,source_preservation=True,prediction_algebra_checked=True,
        corner_accounting_and_contrast_checked=True,scope='small post-hoc diagnostic'))
    print(json.dumps(dict(predictions=frozen,summary=summary),indent=2),flush=True)


if __name__=='__main__':main()
