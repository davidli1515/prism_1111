"""Independent finite-set and closed-form ridge checks; no model queries."""
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent/'results_s42_01'
read = lambda n: json.loads((OUT/n).read_text())
plan, roots, pred, train, labels, summary = map(read, (
    'plan.json', 'roots.json', 'predictions.json', 'training_labels.json', 'test_labels.json', 'summary.json'))
assert hashlib.sha256((OUT/'predictions.json').read_bytes()).hexdigest() == summary['predictions_sha256']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in plan['source_sha256'].items())
seen = set()
for row in plan['train']+plan['test']:
    c = row['root']
    expected = {(f, a) for f, a in product((8,16,32,64),(2,4,8,16,32))
                if f >= c['fetchBufferSize'] and a >= c['l1i_assoc'] and a <= c['l1_size']
                and (f,a) != (c['fetchBufferSize'],c['l1i_assoc'])}
    assert {(x['fetchBufferSize'], x['l1i_assoc']) for x in row['candidates']} == expected
    for x in row['candidates']:
        assert all(x[k] == c[k] for k in c if k not in ('fetchBufferSize','l1i_assoc'))
    g = (row['bench'], tuple(sorted((k,v) for k,v in c.items() if k not in ('fetchBufferSize','l1i_assoc'))))
    assert g not in seen
    seen.add(g)
assert len(labels) == 3 and sum(len(x['records']) for x in labels) == 9
for planned, row in zip(plan['test'], labels):
    assert planned['candidates'] == [x['config'] for x in row['records']]
    assert abs(row['gain']-max(0.,1-min(x['cpi'] for x in row['records'])/row['root_cpi'])) < 1e-12
n = len(train)
y = np.array([x['gain'] for x in train])
s = np.array([np.concatenate(r['blocks'][:3]) for r in roots])
m = np.array([r['blocks'][3] for r in roots])
shuffled = np.r_[m[:n][pred['train_permutation']], m[n:][pred['test_permutation']]]
cap = np.array([1-1/(min(x['root'][k] for k in ('fetchWidth','decodeWidth','dispatchWidth','commitWidth'))*x['root_cpi']) for x in plan['test']])
truth = np.array([x['gain'] for x in labels])
errors = {}
for name, x in (('scalar',s),('mscs',np.c_[s,m]),('shuffled',np.c_[s,shuffled])):
    mu=x[:n].mean(0); centered=x[:n]-mu
    # Dual ridge with intercept, alpha=1; independent of sklearn and run.py.
    coeff=np.linalg.solve(centered@centered.T+np.eye(n),y-y.mean())
    independent=np.clip((x[n:]-mu)@centered.T@coeff+y.mean(),0,cap)
    err=float(np.max(abs(independent-pred['predictions'][name])))
    assert err < 1e-10
    assert abs(np.mean(abs(independent-truth))-summary['methods'][name]['mae']) < 1e-10
    errors[name]=err
result={'passed':True,'train_sets':n,'test_sets':3,'test_candidates':9,
        'manual_ridge_max_disagreement':errors,'source_hashes_verified':len(plan['source_sha256'])}
with (OUT/'validation.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
