"""Combine independent baseline/local replays and check median root selection."""
from pathlib import Path
import numpy as np
from local_support import config_key,legal
from fresh_local_validate import validate as local_validate,sha,read,close
from fresh_controls_validate import validate as controls_validate,pareto_mask


def validate(out):
    out=Path(out);plan=read(out/'plan.json')
    inventory=read(out/plan['exclusion_file']);assert sha(out/plan['exclusion_file'])==plan['exclusion_sha256']
    used={(r['bench'],config_key(r['config'])) for r in inventory['exposed']}
    selected=[]
    for ctx,s in zip(plan['contexts'],plan['root_selection']):
        archive=read(s['archive']);points=np.c_[archive['cpis'],archive['areas']]
        front=np.flatnonzero(pareto_mask(points));front=front[np.argsort(points[front,1],kind='stable')]
        assert front.tolist()==s['front_archive_indices']
        target=(len(front)-1)//2
        valid=[legal(archive['configs'][int(i)]) and (ctx['bench'],config_key(archive['configs'][int(i)])) not in used for i in front]
        rank=min((j for j,ok in enumerate(valid) if ok),key=lambda j:(abs(j-target),j))
        assert target==s['target_rank'] and rank==s['selected_rank']==ctx['rank']
        assert valid==s['eligible_by_rank'];i=int(front[rank]);assert i==ctx['archive_index']
        assert ctx['root']==archive['configs'][i] and (ctx['bench'],config_key(ctx['root'])) not in used
        close(ctx['root_cpi'],archive['cpis'][i]);close(ctx['root_area'],archive['areas'][i])
        close(ctx['area_cap'],1.10*ctx['root_area'])
        selected.append(dict(context=ctx['id'],target=target,selected=rank,passed=True))
    controls=controls_validate(out);local=local_validate(out)
    total=controls['total_candidate_queries']+local['total_candidate_queries']
    assert total<=plan['max_candidate_verifications']
    assert total==sum(w['counts']['candidate_forward_calls'] for w in read(out/'summary.json')['walks'])
    return dict(passed=True,total_candidate_queries=total,selection=selected,
                controls=controls,local=local,policy_unchanged=True)
