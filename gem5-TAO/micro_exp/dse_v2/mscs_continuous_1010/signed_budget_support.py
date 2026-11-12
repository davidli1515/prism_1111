"""Exchange enumeration is independent of forecast scores and performance."""
from secant_support import config_key,compute_w_min
from core import legal


def frozen_for(bench,audit):
    names=['l2_size','l2_assoc']
    if bench in audit:
        r=audit[bench]
        names += [k for k,v in r['groups'].items() if v['all_counts_present'] and
                  v['all_units_mapped'] and v['zero_issued_runs']==r['runs']]
    return names


def exchange_pairs(root,pool,seen):
    used=set(seen)|{config_key(root)};out=[]
    basis=[i for i,a in enumerate(pool) if a['kind']=='single_resource']
    for off,i in enumerate(basis):
        a=pool[i]
        for j in basis[off+1:]:
            b=pool[j]
            if a['parameter']==b['parameter'] or a['sign']==b['sign']:continue
            c=dict(root);c[a['parameter']]=a['config'][a['parameter']];c[b['parameter']]=b['config'][b['parameter']]
            key=config_key(c)
            if key in used or not legal(c):continue
            assert compute_w_min(c)==compute_w_min(root)
            used.add(key);out.append(dict(indices=[i,j],config=c,kind='opposite_resource_exchange',
                edits={k:dict(before=root[k],after=c[k]) for k in root if c[k]!=root[k]}))
    return out
