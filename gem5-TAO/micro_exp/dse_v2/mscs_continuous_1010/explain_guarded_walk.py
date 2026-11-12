"""Per-step, auditable component accounts; never label proxy agreement as gem5."""
import csv
import json
import numpy as np
from secant_support import HERE,read,sha,save,compute_w_min
from width_priority_validate import cpi


def account(m,width):
    m=np.asarray(m,dtype=float);stages=cpi(m,width)
    by_stage=(stages[...,None]*m).mean(0)
    np.testing.assert_allclose(by_stage.sum(-1),stages.mean(0),atol=1e-8)
    return by_stage


def main():
    src=HERE/'guarded_greedy_s42_01';out=HERE/'guarded_explanations_01';out.mkdir(exist_ok=False)
    assert read(src/'validation.json')['passed']
    rows=[];sources={};fu=read(HERE/'unused_fu_audit_01/summary.json')['by_workload']
    for path in sorted((src/'walks').glob('*/mscs_guard/decision_*.json')):
        d=read(path);rpath=path.with_name(path.name.replace('decision','round'));r=read(rpath)
        i=d['choice'];assert r['resolved'];action=d['candidates'][i];truth=src/'truth'/d['bench']/f"{r['trial']['label_id']}.npz"
        use_memory=bool(d['withdrawal'][i] and d['memory_mscs_scores'][i]>d['raw_mscs_scores'][i])
        with np.load(d['prediction_file']) as z:
            root=z['root_m'];pred=z['memory__predicted_m'][i] if use_memory else z['predicted_m'][i]
        with np.load(truth) as z:actual=z['m']
        w=compute_w_min(action['config']);w0=compute_w_min(d['root'])
        ac0=account(root,w0);acp=account(pred,w);act=account(actual,w)
        predicted=float(acp.sum(-1).mean());observed=float(act.sum(-1).mean())
        np.testing.assert_allclose(predicted,d['mscs_scores'][i],rtol=4e-6)
        np.testing.assert_allclose(observed,r['trial']['cpi'],rtol=4e-6)
        dp=acp-ac0;dt=act-ac0
        known_unused=[]
        if d['bench'] in fu:
            ff=fu[d['bench']]
            known_unused=[k for k,v in ff['groups'].items() if k in action['edits'] and
                v['all_counts_present'] and v['all_units_mapped'] and v['zero_issued_runs']==ff['runs']]
        row=dict(bench=d['bench'],query=d['step']+1,edits=action['edits'],accepted=r['accepted'],
            ranking_response='history' if use_memory else 'root',withdrawal=d['withdrawal'][i],
            root_cpi=d['root_cpi'],predicted_cpi=d['mscs_scores'][i],actual_cpi=r['trial']['cpi'],
            predicted_gain_pct=100*(1-d['mscs_scores'][i]/d['root_cpi']),
            actual_gain_pct=100*(1-r['trial']['cpi']/d['root_cpi']),
            root_accounts_by_stage=ac0.tolist(),predicted_account_delta_by_stage=dp.tolist(),
            actual_account_delta_by_stage=dt.tolist(),predicted_mean_delta=dp.mean(0).tolist(),
            actual_mean_delta=dt.mean(0).tolist(),unused_units_in_archived_ROI=known_unused,
            scope='Full frozen-surrogate response. Stage accounts are alternative views; no causal hardware attribution.')
        rows.append(row)
        for p in (path,rpath,d['prediction_file'],truth):sources[str(p)]=sha(p)
    assert len(rows)==18
    save(out/'records.json',rows)
    pdelta=np.array([r['predicted_mean_delta'][1:] for r in rows]);tdelta=np.array([r['actual_mean_delta'][1:] for r in rows])
    summary=dict(steps=len(rows),accepted=sum(r['accepted'] for r in rows),
        history_scored_selected_moves=sum(r['ranking_response']=='history' for r in rows),
        component_sign_agreement=float(np.mean(np.sign(pdelta)==np.sign(tdelta))),
        component_CPI_MAE=float(np.abs(pdelta-tdelta).mean()),
        moves_touching_archived_unused_units=sum(bool(r['unused_units_in_archived_ROI']) for r in rows),
        components=['Base','FP','BP','SP'],stages=['Decode','Dispatch','Issue','Commit'],
        averaging='Arithmetic mean over 50 slices and four alternative stage accounts; stages are never added as separate elapsed times',
        new_model_queries=0,new_gem5_runs=0,source_sha256=sources)
    save(out/'summary.json',summary)
    with (out/'steps.csv').open('x',newline='') as f:
        fields=['bench','query','edits','accepted','ranking_response','predicted_gain_pct','actual_gain_pct','unused_units_in_archived_ROI']
        fields += [kind+'_'+c for kind in ('predicted','actual') for c in ('Base','FP','BP','SP')]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for row in rows:
            rec={k:row[k] for k in fields if k in row};rec['edits']=json.dumps(rec['edits'],sort_keys=True)
            for kind in ('predicted','actual'):
                rec.update({kind+'_'+c:v for c,v in zip(summary['components'],row[kind+'_mean_delta'])})
            writer.writerow(rec)
    assert all(sha(p)==h for p,h in sources.items())
    save(out/'validation.json',dict(passed=True,selected_ranking_response_matches_score=True,
        component_accounts_sum_to_stage_CPI=True,all18_paid_steps_included=True,sources_unchanged=True,new_queries=0))
    print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},indent=2))


if __name__=='__main__':main()
