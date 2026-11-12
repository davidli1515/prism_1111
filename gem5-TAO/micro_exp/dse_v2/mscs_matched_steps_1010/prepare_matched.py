"""Prediction-only selection; a separate process evaluates after plan is frozen."""
import json
from matched_support import *


def main():
    OUT.mkdir(exist_ok=False)
    hashes = {}; selected = []; survey = []; contexts = []; required = {}
    checkpoint_hash = sha(DEFAULT_CKPT)
    assert checkpoint_hash == '8780518a8b66c766873a6f268e5a9d5307332ae8c61edb6860fbc0eae873bdf2'
    for directory in SOURCES:
        plan = read(directory/'plan.json')
        for p, h in plan['source_sha256'].items():
            assert sha(p) == h, p
        hashes.update(plan['source_sha256'])
        hashes[str(directory/'plan.json')] = sha(directory/'plan.json')
        for ctx in plan['contexts']:
            ctx = dict(ctx, source_directory=str(directory))
            contexts.append(ctx); found = set()
            base = directory/'walks'/ctx['id']/'local'
            for p in list(base.glob('round_*.json'))+[directory/'inputs'/f"{ctx['id']}.json"]:
                hashes[str(p)] = sha(p)
            for path in sorted(base.glob('decision_*.json')):
                d = read(path); hashes[str(path)] = sha(path)
                snap = directory/d['prediction_file']
                assert sha(snap) == d['prediction_sha256']
                hashes[str(snap)] = sha(snap)
                for kind in KINDS:
                    ids = pool_indices(d, kind)
                    row = dict(context=ctx['id'], step=d['step'], pool=kind,
                               decision_file=str(path), pool_indices=ids, selected=False)
                    if len(ids) < 2:
                        row['reason'] = 'fewer_than_two'; survey.append(row); continue
                    pairs = choices(d, ids); row['chosen'] = pairs
                    disagreements = [a for a in ARMS[1:] if set(pairs[a]) != set(pairs['scalar'])]
                    local_change = set(pairs['local']) != set(pairs['hybrid'])
                    row.update(disagreements=disagreements, local_vs_hybrid=local_change)
                    triggers = []
                    if disagreements and kind not in found:
                        triggers.append('first_scalar_disagreement_'+kind); found.add(kind)
                    if kind == 'full' and local_change and 'local_change' not in found:
                        triggers.append('first_local_vs_hybrid'); found.add('local_change')
                    if triggers:
                        row['selected'] = True
                        cell = dict(row, cell_id=f"{ctx['id']}_r{d['step']+1}_{kind}",
                                    triggers=triggers, root=d['root'], root_cpi=d['root_cpi'],
                                    radius=d['radius'], bench=ctx['bench'], area_cap=ctx['area_cap'])
                        cell['label_ids'] = {str(i): label_id(ctx['bench'], d['candidates'][i]['config'])
                                             for i in sorted(set(sum(pairs.values(), [])))}
                        selected.append(cell)
                        for i, lid in cell['label_ids'].items():
                            a = d['candidates'][int(i)]
                            rec = required.setdefault(lid, dict(id=lid, bench=ctx['bench'],
                                config=a['config'], cells=[], root_cpis=[]))
                            rec['cells'].append(cell['cell_id']); rec['root_cpis'].append(d['root_cpi'])
                    survey.append(row)
    assert len(contexts) == 5 and len(selected) <= 25 and len(required) <= 200
    for p in list(HERE.glob('*.py'))+[HERE/'PROTOCOL.txt']:
        hashes[str(p)] = sha(p)
    result = dict(seed=42, arms=ARMS, candidate_budget_per_arm=2,
                  scope='prediction-selected matched-state historical mechanism audit',
                  checkpoint=str(DEFAULT_CKPT), checkpoint_sha256=checkpoint_hash,
                  contexts=contexts, survey=survey, cells=selected,
                  required_labels=list(required.values()), source_sha256=hashes,
                  phase='frozen before loading historical trial labels or new candidate evaluation')
    save(OUT/'plan.json', result)
    print(json.dumps(dict(cells=len(selected), distinct_required_labels=len(required),
                          selected=[c['cell_id'] for c in selected], plan_sha256=sha(OUT/'plan.json')), indent=2))


if __name__ == '__main__':
    main()
