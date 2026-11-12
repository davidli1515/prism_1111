#!/usr/bin/env python3
"""Independent saved-output audit: no model inference and no selector imports."""
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'results_s42_01'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hv_by_strips(points, ref):
    # Independent rectangle-union integration, without Pareto filtering.
    p = [x for x in points if x[0] < ref[0] and x[1] < ref[1]]
    xs = sorted({x[0] for x in p} | {ref[0]})
    return sum((r-l)*(ref[1]-min(x[1] for x in p if x[0] <= l))
               for l, r in zip(xs[:-1], xs[1:]))


def main():
    started = time.monotonic()
    plan, decisions, summary = [read(OUT/n) for n in ('plan.json', 'decisions.json', 'summary.json')]
    assert digest(OUT/'decisions.json') == summary['decisions_sha256']
    assert all(digest(p) == h for p, h in plan['source_sha256'].items())
    assert all(digest(p) == h for p, h in summary['descendant_CPI_reuse_source_sha256'].items())
    all_keys, rows = set(), []
    negative_areas = []
    for p, d in zip(plan['contexts'], decisions['contexts']):
        r = read(OUT/f"audit_{d['id']}.json")
        archive = plan['archives'][d['bench']]
        src = read(archive['source'])
        # Independently reconstruct the BO Pareto set and root selection.
        pts = list(zip(src['cpis'], src['areas']))
        frontier = [i for i, a in enumerate(pts)
                    if not any(b[0] <= a[0] and b[1] <= a[1] and b != a for b in pts)]
        frontier.sort(key=lambda i: (pts[i][1], i))
        label = d['id'].rsplit('_', 1)[1]
        rank = {'low': 0, 'middle': (len(frontier)-1)//2, 'high': len(frontier)-1}[label]
        assert d['root_index'] == frontier[rank]
        c = d['root']
        expected = {(f, a) for f, a in product((8, 16, 32, 64), (2, 4, 8, 16, 32))
                    if f >= c['fetchBufferSize'] and a >= c['l1i_assoc'] and a <= c['l1_size']
                    and (f, a) != (c['fetchBufferSize'], c['l1i_assoc'])}
        assert {(x['fetchBufferSize'], x['l1i_assoc']) for x in d['candidates']} == expected
        assert [x['config'] for x in r['labels']] == d['candidates'] == p['candidates']
        for x, known in zip(r['labels'], d['known_at_snapshot']):
            assert all(x['config'][k] == c[k] for k in c if k not in ('fetchBufferSize', 'l1i_assoc'))
            assert known == (x['config'] in src['configs'])
            k = (d['bench'], tuple(sorted(x['config'].items())))
            assert k not in all_keys  # Savings cannot double-count overlapping sets.
            all_keys.add(k)
        f, ref = archive['front'], archive['reference']
        cp = [[x['cpi'], x['area']] for x in r['labels'] if x['area'] is not None]
        independent_hvi = hv_by_strips(f+cp, ref)-hv_by_strips(f, ref)
        assert abs(independent_hvi-r['region_hvi']) < 1e-9
        m = np.array(d['mscs'], dtype=np.float32).reshape(-1, 4, 4)
        width = min(c[k] for k in ('fetchWidth', 'decodeWidth', 'dispatchWidth', 'commitWidth'))
        ci = np.minimum(1/(np.maximum(m[:, :, 0], .01)*width+1e-8), 100.).mean(1).astype(float)
        bounds = {'scalar_width': 1/width,
                  'mscs_decode': max(1/width, np.mean(ci*(1-m[:, 0, 1].astype(float)))),
                  'mscs_stage_max': max(1/width, np.mean(ci*(1-m[:, :, 1].max(1).astype(float))))}
        amin = min(d['areas']) if d['areas'] and all(x is not None for x in d['areas']) else 0.
        assert amin == d['area_lower']
        for method, lower in bounds.items():
            assert abs(lower-d['bounds'][method]) < 1e-10
            prune = any(q[0] <= lower and q[1] <= amin for q in f)
            assert prune == d['methods'][method]['prune'] == r['methods'][method]['prune']
            v = [j for j, x in enumerate(r['labels']) if x['cpi'] < lower-1e-6*max(1., abs(lower), x['cpi'])]
            assert v == r['methods'][method]['violations']
        eligible = [q[0] for q in f if q[1] <= amin]
        threshold = min(eligible) if eligible else None
        root_cpi = d['root_point'][0]
        rows.append({'id': d['id'], 'n': len(cp), 'root_cpi': root_cpi,
                     'mscs_lower': float(bounds['mscs_decode']), 'required_lower_to_prune': threshold,
                     'best_actual_cpi': r['best_actual_cpi'], 'region_hvi': r['region_hvi'],
                     'mscs_max_gain_pct': 100*(1-bounds['mscs_decode']/root_cpi),
                     'actual_best_gain_pct': 100*(1-r['best_actual_cpi']/root_cpi) if cp else None,
                     'oracle_corner_prunable': r['oracle_corner_prunable'],
                     'minimum_area_increment': r['area_increment_min']})
        if cp and amin < d['root_point'][1]:
            child = min(r['labels'], key=lambda x: x['area'])
            negative_areas.append((c, d['root_point'][1], child['config'], child['area']))
    assert len(all_keys) == summary['candidates'] == 18
    # A nonmonotonic McPAT result is material to area-bound correctness. Recheck
    # the root and child serially with completely fresh, isolated XML files.
    sys.path.insert(0, str(HERE.parent.parent/'dse_exp'))
    from area_model import compute_area
    rechecks = []
    for c, expected, child, target in negative_areas:
        for config, old in ((c, expected), (child, target)):
            with tempfile.TemporaryDirectory(prefix='area_recheck_', dir=OUT) as work:
                got = compute_area(config, work_dir=work)
            assert abs(got-old) < 1e-6
            rechecks.append({'config': config, 'saved_area': old, 'fresh_serial_area': got})
    result = {'independent_checks_passed': True, 'checked_regions': len(rows),
              'checked_candidates': len(all_keys), 'area_rechecks': rechecks,
              'rows': rows, 'seconds': time.monotonic()-started}
    with (OUT/'validation.json').open('x') as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
