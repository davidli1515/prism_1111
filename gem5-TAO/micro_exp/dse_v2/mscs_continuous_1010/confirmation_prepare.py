"""Scalar archive geometry chooses new roots, never their new search outcomes."""
from pathlib import Path
import sys
import numpy as np
from secant_support import HERE,ROOT,read,sha,save,config_key
sys.path.insert(0,str(ROOT/'mscs_fresh_disagreement_1010'))
from fresh_disagreement_support import configs_in,bench_of,legal,pareto_mask

OUT=HERE/'confirmation_s42_01'
BENCHES=('602.gcc_s','605.mcf_s','623.xalancbmk_s','641.leela_s')


def main():
    OUT.mkdir(exist_ok=False)
    dev=HERE/'quadratic_development_s42_01'
    assert read(dev/'summary.json')['gate_passed'] and read(dev/'validation_replay_v2.json')['passed']
    sources=dict(read(dev/'plan.json')['source_sha256'])
    assert all(sha(p)==h for p,h in sources.items())
    exposed,unknown,inventory=set(),set(),{}
    directories=sorted(p for p in ROOT.glob('mscs_*_1010') if p.is_dir())
    for directory in directories:
        for path in sorted(directory.rglob('*.json')):
            if OUT in path.parents:continue
            hint=next((bench_of(s) for s in path.parts if bench_of(s)),None)
            f,u=configs_in(read(path),hint);exposed.update(f);unknown.update(u);inventory[str(path)]=sha(path)
    exposed.update((b,c) for c in unknown for b in BENCHES)
    sources.update(inventory)
    save(OUT/'exclusions.json',dict(source_sha256=inventory,
        unknown_context_configs=[dict(c) for c in sorted(unknown)],
        exposed=[dict(bench=b,config=dict(c)) for b,c in sorted(exposed)]))
    contexts=[];selection=[]
    for bench in BENCHES:
        file=ROOT/'mscs_guard_v5/results_full_01'/bench/'s42/initial.json';archive=read(file);sources[str(file)]=sha(file)
        points=np.c_[archive['cpis'],archive['areas']];front=np.flatnonzero(pareto_mask(points))
        front=front[np.argsort(points[front,1],kind='stable')];target=(len(front)-1)//2
        eligible=[legal(archive['configs'][int(i)]) and (bench,config_key(archive['configs'][int(i)])) not in exposed for i in front]
        rank=min((j for j,yes in enumerate(eligible) if yes),key=lambda j:(abs(j-target),j),default=None)
        selection.append(dict(bench=bench,archive=str(file),front=front.tolist(),target=target,eligible=eligible,selected_rank=rank))
        if rank is None:continue
        i=int(front[rank]);c=archive['configs'][i]
        contexts.append(dict(id=f'{bench}_qconfirm_idx{i}',bench=bench,archive_index=i,area_rank=rank,
            root=c,root_cpi=archive['cpis'][i],root_area=archive['areas'][i],
            area_cap=1.10*archive['areas'][i],area_sources=archive['area_sources']))
    for file in list(HERE.glob('confirmation_*.py'))+[HERE/'confirmation_protocol.txt',HERE/'quadratic_validate_replay_v2.py']:
        sources[str(file)]=sha(file)
    save(OUT/'root_plan.json',dict(contexts=contexts,selection=selection,source_sha256=sources,
        exclusions_sha256=sha(OUT/'exclusions.json'),scope='four predetermined fresh configurations for this development series'))
    print(__import__('json').dumps(dict(roots=[c['id'] for c in contexts],excluded=len(exposed),unfilled=4-len(contexts))),flush=True)


if __name__=='__main__':main()
