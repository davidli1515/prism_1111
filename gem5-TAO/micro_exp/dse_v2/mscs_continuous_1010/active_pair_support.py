"""A single greedy proposal; no access to performance labels or other histories."""
import torch
from secant_support import compute_w_min
from pair_priority_support import dense_singles as original_singles,pair_universe
from width_priority_support import predict
from lazy_pair_support_v2 import choose_pair


INACTIVE_PARAMETERS=frozenset(('l2_size','l2_assoc'))


def dense_singles(root,seen):
    # Filter by complete edits rather than only the primary parameter label.
    return [a for a in original_singles(root,seen)
            if all(a['config'][k]==root[k] for k in INACTIVE_PARAMETERS)]


def prepare(local,cache,root,radius,seen,joint,area_get,cap):
    singles=[a for a in dense_singles(root,seen) if a['stride']<=radius]
    assert singles
    xs=torch.stack([cache.get(a['config']) for a in singles]);width=compute_w_min(root)
    de=torch.einsum('nskd,cnd->cnsk',local.gell,xs-local.x[None])
    pairs=[];proposed={};traces={};pair_scores={};info=[];extra=[];pool=list(singles);resolved=True
    if joint:
        positive=[i for i,a in enumerate(singles) if a['kind']=='single_resource' and a['sign']==1]
        pairs=pair_universe(root,singles,positive,seen)
        if pairs:
            ids=torch.tensor([a['indices'] for a in pairs]);pair_de=de[ids[:,0]]+de[ids[:,1]]
            pp=predict(local.ell,pair_de,[width]*len(pairs))
            for kind in ('scalar','mscs'):
                pair_scores[kind]=pp[kind+'_cpi'].tolist()
                rec=choose_pair(pairs,pp[kind+'_cpi'],singles,area_get,cap);traces[kind]=rec;proposed[kind]=rec['choice']
                resolved &= rec['resolved']
            for kind,i in proposed.items():
                if i is None:continue
                a=pairs[i]
                if any(v['config']==a['config'] for v in pool):continue
                x=cache.get(a['config']);delta=x-local.x;j,k=a['indices']
                info.append(dict(generator=kind,pair_index=i,pool_index=len(pool),input_chord_error=float(torch.linalg.norm(delta-(xs[j]-local.x)-(xs[k]-local.x)))))
                extra.append(torch.einsum('nskd,nd->nsk',local.gell,delta));pool.append(a)
    full_de=torch.cat([de,torch.stack(extra)]) if extra else de
    widths=[compute_w_min(a['config']) for a in pool];pred=predict(local.ell,full_de,widths)
    arrays=dict(root_m=local.m,root_ell=local.ell,logodds_slope=full_de,predicted_m=pred['predicted_m'],
        scalar_stage=pred['scalar_stage'],mscs_stage=pred['mscs_stage'])
    record=dict(candidates=pool,widths=widths,singles_count=len(singles),pairs=pairs,pair_scores=pair_scores,
        proposed_pairs=proposed,pair_traces=traces,full_pair_info=info,resolved=bool(resolved),
        scalar_scores=pred['scalar_cpi'].tolist(),mscs_scores=pred['mscs_cpi'].tolist())
    return record,arrays
