"""Finite component costs are used only when ranking resource withdrawals."""
import numpy as np

ARMS=('scalar_raw','scalar_memory','scalar_guard','scalar_upfirst','mscs_directional')


def ranking(root,root_cpi,candidates,raw_scalar,memory_scalar,raw_mscs,arm):
    withdrawal=np.array([any(a['config'][k]<root[k] for k in root) for a in candidates],dtype=bool)
    raw=np.asarray(raw_scalar,dtype=float);memory=np.asarray(memory_scalar,dtype=float)
    scores=raw.copy();blocked=np.zeros(len(candidates),dtype=bool)
    if arm=='scalar_memory':scores=memory
    elif arm=='scalar_guard':scores=np.where(withdrawal,np.maximum(raw,memory),raw)
    elif arm=='scalar_upfirst':blocked=withdrawal.copy()
    elif arm=='mscs_directional':scores=np.where(withdrawal,np.asarray(raw_mscs,dtype=float),raw)
    else:assert arm=='scalar_raw'
    order=sorted(range(len(candidates)),key=lambda i:(bool(blocked[i]),float(scores[i]),i))
    return order,blocked.tolist(),withdrawal.tolist()
