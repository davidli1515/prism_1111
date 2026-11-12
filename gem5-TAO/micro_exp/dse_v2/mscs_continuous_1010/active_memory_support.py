"""Unchanged raw pair proposals, interpolation only for final candidate rank."""
import torch
from active_pair_support import prepare
from paid_centers_support import predict


def prepare_memory(local,centers,cache,root,radius,seen,joint,area_get,cap):
    record,arrays=prepare(local,cache,root,radius,seen,joint,area_get,cap)
    xs=torch.stack([cache.get(a['config']) for a in record['candidates']])
    predictions,extra=predict(centers,xs,record['widths'])
    for kind in ('scalar','mscs'):
        record['raw_'+kind+'_scores']=record[kind+'_scores']
        record[kind+'_scores']=predictions[kind+'_blend'].tolist()
    arrays.update({'memory__'+k:v for k,v in extra.items()})
    return record,arrays
